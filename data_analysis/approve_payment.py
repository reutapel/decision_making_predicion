import os
import pandas as pd


# base_directory = os.path.abspath(os.curdir)
# data_directory = os.path.join(base_directory, 'results')
# date_directory = 'text_exp_2_tests'

base_directory = os.path.join('/Users', 'reutapel', 'Downloads')

number_of_rounds = 10


def define_status():
    results = pd.read_excel(os.path.join(base_directory, 'text_exp_deterministic_2020-03-08.xlsx'))  #, sheetname='data_to_use')
    results.columns = results.columns.str.replace(r".", "_")

    results = results.assign(status='')
    # adding pair_id
    if 'pair_id' not in results.columns:
        results['pair_id'] = results['session_code'] + '_' + results['group_id_in_subsession'].map(str)

    answered_correct_first_test = results.loc[(results.player_intro_test.str.lower() == 'sdkot') &
                                              (results.subsession_round_number == 1)].participant_code.unique()
    not_answered_correct_first_test = results.loc[(results.player_intro_test.str.lower() != 'sdkot') &
                                                  (~results.player_intro_test.isnull()) &
                                                  (results.subsession_round_number == 1)].participant_code.unique()
    results.loc[results.participant_code.isin(answered_correct_first_test), 'first_test_status'] = 1
    results.loc[results.participant_code.isin(not_answered_correct_first_test), 'first_test_status'] = 0
    # define status
    participants_wait = results.loc[(results.participant_payoff <= 0.9) & (results.player_intro_test.isnull()) &
                                    (results.subsession_round_number == 1) &
                                    (~results.participant_mturk_worker_id.isnull())].participant_code.unique()
    results.loc[results.participant_code.isin(participants_wait), 'status'] = 'wait'
    results.loc[(results.participant_payoff == 0) & (results.player_intro_test.isnull()) &
                (~results.participant_mturk_worker_id.isnull()), 'status'] = 'drop_timeout'
    pass_first_test_partner_not = results.loc[(results.first_test_status == 1) & (results.player_name.isnull()) &
                                              (results.subsession_round_number == 1)].participant_code.unique()
    results.loc[(results.participant_code.isin(pass_first_test_partner_not)) &
                (results.participant__current_page_name != 'AfterIntroTest'), 'status'] = 'pass_first_test_partner_not'
    results.loc[(results.participant_code.isin(pass_first_test_partner_not)) &
                (results.participant__current_page_name == 'AfterIntroTest'), 'status'] = 'pass_first_test_partner_drop'
    results.loc[(results.first_test_status == 0), 'status'] = 'failed_first_test'
    results.loc[(results.group_receiver_passed_test == 0) & (results.player_id_in_group == 2) &
                (results.first_test_status == 1), 'status'] =\
        'dm_failed_second_no_pay'
    results.loc[(results.group_receiver_passed_test == 0.5) & (results.player_id_in_group == 2) &
                (results.first_test_status == 1), 'status'] =\
        'dm_half_passed_second_no_bonus'
    results.loc[(results.group_receiver_passed_test == 1) & (results.player_id_in_group == 2) &
                (results.first_test_status == 1), 'status'] =\
        'dm_passed_second'
    results.loc[(results.participant__current_page_name == 'Test') & (results.player_id_in_group == 2) &
                (results.first_test_status == 1), 'status'] = 'dm_not_took_second_test'
    results.loc[(results.first_test_status == 1) & (~results.participant_code.isin(pass_first_test_partner_not))&
                (results.player_id_in_group == 1), 'status'] = 'expert_pass_first_test'

    return results


def check_timeouts(results):
    play_pay_data_timeout_sum = results.groupby(by='participant_code').agg({'group_receiver_timeout': 'sum',
                                                                            'group_sender_timeout': 'sum'})

    play_pay_data_timeout_sum.columns = ['sum_group_receiver_timeout', 'sum_group_sender_timeout']
    results = results.merge(play_pay_data_timeout_sum, how='outer', left_on='participant_code', right_index=True)

    return results


def define_payment(results):
    results = results.assign(payment='')
    results.loc[results.status.isin(['wait', 'pass_first_test_partner_not', 'dm_half_passed_second_no_bonus',
                                     'dm_passed_second', 'expert_pass_first_test']), 'payment'] =\
        results.participant_payoff
    results.loc[results.status.isin(['failed_first_test', 'dm_failed_second_no_pay']), 'payment'] = 0
    # if they had more than number_of_rounds/2 timeouts we will not pay them
    results.loc[(results.status.isin(['dm_half_passed_second_no_bonus', 'dm_passed_second', 'expert_pass_first_test']))
                & (((results.sum_group_receiver_timeout > number_of_rounds/2) & (results.player_id_in_group == 2)) |
                   ((results.sum_group_sender_timeout > number_of_rounds / 2) & (results.player_id_in_group == 1))),
                'payment'] = 0
    results.loc[results.status.isin(['pass_first_test_partner_drop']), 'payment'] = results.participant_payoff

    player_intro_test = results.loc[results.subsession_round_number == 1][['player_intro_test', 'participant_code']]
    results = results.drop('player_intro_test', axis=1)
    results = results.merge(player_intro_test, on='participant_code')

    results = results.loc[results.subsession_round_number == number_of_rounds]
    results = results[['participant_code', 'participant_mturk_worker_id', 'participant_mturk_assignment_id', 'pair_id',
                       'session_code', 'status', 'participant_payoff', 'payment', 'player_id_in_group',
                       'player_intro_test', 'first_test_status', 'player_dm_test_chosen_review_1',
                       'player_dm_test_chosen_review_2', 'player_dm_test_not_chosen_review_1',
                       'player_dm_test_not_chosen_review_2', 'group_id_in_subsession', 'sum_group_receiver_timeout',
                       'sum_group_sender_timeout', 'group_receiver_passed_test']]

    results.to_csv(os.path.join(base_directory, 'text_exp_data_payment.csv'))

    return


if __name__ == '__main__':
    outer_results = define_status()
    outer_results = check_timeouts(outer_results)
    define_payment(outer_results)




