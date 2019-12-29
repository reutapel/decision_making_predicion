import os
import logging
import tempural_analysis.utils as utils
import tempural_analysis.create_data as create_data
from datetime import datetime
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from tempural_analysis.majority_rule import MajorityRule
import itertools
import tempural_analysis.execute_evaluate as execute_evaluate
from tempural_analysis.majority_per_problem import MajorityPerProblem
from tempural_analysis.majority_last_trials import MajorityLastTrials
from xgboost import XGBClassifier, XGBRegressor
from tempural_analysis.learn_majority_last_trials import LearnMajorityLastTrials
import copy


def main():
    base_directory = os.path.abspath(os.curdir)
    data_directory = os.path.join(os.path.abspath(os.path.join(base_directory, os.pardir)), 'results')
    exp_directory = 'one_player'
    is_backward = False
    sample_participants = [True, 60]
    if is_backward:
        backward_appen = 'backward'
    else:
        backward_appen = 'grid'
    # dict for the directories file names. dir_name: [file name, if not_agent_data(2 players exp), number_of_trials,
    # known_unknown_exp]
    directory_files_dict = {
        'agent_mturk': ['all_data_after_4_1_and_data_to_use_before.xlsx', False, 50, False],
        'one_player': ['first_results.xlsx', False, 50, False],
        'agent_no_payment': ['agent_no_payment.xlsx', False, 50, False],
        'all': ['results_payments_status.csv', True, 50, False],
        'second_agent_exp': ['mturk_ag_second_exp.xlsx', False, 60, True],
        'all_agents': ['concat_data_27_02.xlsx', False, 50, False],
    }
    not_agent_data = directory_files_dict[exp_directory][1]
    num_trials = directory_files_dict[exp_directory][2]
    known_unknown_exp = directory_files_dict[exp_directory][3]
    appendix = 'group' if not_agent_data else 'player'

    # create folder for experiment
    exp_directory_path = utils.set_folder(exp_directory, 'classifier_results')
    experiment_path = utils.set_folder(datetime.now().strftime(backward_appen + '_%d_%m_%Y_%H_%M'), exp_directory_path)

    file_name = datetime.now().strftime('LogFile_temporary_role_effect.log')
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S',
                        filename=os.path.join(experiment_path, file_name),
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    ##################################################################################################

    folds = 5  # if folds = 1, run train-test,
               # else - run cross-validation with relevant number of folds

    """The base features are the all features we will use that are in the original data,
    and the add_feature_list are the features we want to add each time.
    Use receiver_timeout and lottery_result only on time t'<t and only in temporal analysis"""
    orign_data_features = [appendix + '_lottery_result', appendix + '_receiver_timeout', appendix + '_x_lottery',
                           appendix + '_y_lottery', appendix + '_p_lottery', 'subsession_round_number', 'player_age',
                           'player_is_student', 'payed', 'participant_code']
    columns_not_features = ['participant_code']  # 'player_expert_type', 'player_id_in_group']
    add_feature_list = [
        # [],
        # [appendix + '_sender_answer'],
        # [appendix + '_receiver_choice'],
        [appendix + '_receiver_choice', appendix + '_sender_answer']
    ]
    all_add_features_lists = list(set(itertools.chain.from_iterable(add_feature_list)))

    """The label we want to predict"""
    label_dict = {
        appendix + '_receiver_choice': ['DM chose SQ', 'DM chose gamble'],
    }
    # data
    personal_features = ['player_age', 'player_is_student', 'payed', 'male', 'female']

    # two_players_features
    if not_agent_data:
        orign_data_features += [appendix + '_sender_timeout']
        label_dict[appendix + '_sender_answer'] = ['regression results']
        # columns_not_features += ['pair_id']

    """The size of window to take as the features of sample at time t, 0- for no temporal classifier"""
    window_size_list = [5, 1, 10]

    model_dict = {appendix + '_receiver_choice':  # classification models:
                  [LearnMajorityLastTrials(Perceptron()), LearnMajorityLastTrials(XGBClassifier()),
                   XGBClassifier(max_depth=5), MajorityLastTrials(), RandomForestClassifier(), MajorityPerProblem()],
                  appendix + '_sender_answer':  # regression models
                  [RandomForestRegressor(), Perceptron(), SGDRegressor(), PassiveAggressiveRegressor(), SVR(),
                   XGBRegressor()]
                  }
    # model_dict = {appendix + '_receiver_choice':  # classification models:
    #               [MajorityRule()],
    #               appendix + '_sender_answer':  # regression models
    #               [MajorityRule()]
    #               }

    data_file_path = os.path.join(data_directory, exp_directory, directory_files_dict[exp_directory][0])
    ##################################################################################################
    # load data and create features
    create_data_obj = create_data.CreateData()
    create_data_obj.load_data(data_file_path, not_agent_data=not_agent_data)
    create_data_obj.create_features_label(list(label_dict.keys()), orign_data_features, all_add_features_lists,
                                          appendix, sample_participants=sample_participants)

    # evaluate_backward_search
    if is_backward:
        backward_add_features_list = [feature for feature in create_data_obj.base_features + all_add_features_lists]
        backward_add_features_list.append(None)
        candidates_features = copy.deepcopy(backward_add_features_list)
        candidates_features.remove('participant_code')
        execute_evaluate.evaluate_backward_search(data=create_data_obj.features_labels,
                                                  base_features=backward_add_features_list,
                                                  candidates_features=candidates_features,
                                                  window_size_list=window_size_list, label_dict=label_dict,
                                                  num_folds=folds, model_dict=model_dict,
                                                  classifier_results_dir=experiment_path, appendix=appendix,
                                                  personal_features=personal_features)

    # evaluate_grid_search
    else:
        # base_features = ['participant_code', 'subsession_round_number']
        # add_feature_list = [feature for feature in create_data_obj.base_features + all_add_features_lists if
        #                     feature not in base_features]
        execute_evaluate.evaluate_grid_search(data=create_data_obj.features_labels,
                                              base_features=create_data_obj.base_features,
                                              add_feature_list=add_feature_list, window_size_list=window_size_list,
                                              label_dict=label_dict, num_folds=folds, model_dict=model_dict,
                                              classifier_results_dir=experiment_path, appendix=appendix,
                                              personal_features=personal_features)

    logging.info('Done!')


if __name__ == '__main__':
    main()
