from language_prediction.train_test_models import *


def main():
    numbers_columns = ['prev_payoff', 'prev_result_low', 'prev_result_med1', 'prev_result_high',
                       'prev_expected_dm_payoff_low', 'prev_expected_dm_payoff_high',
                       'history_lottery_result', 'history_decisions', 'history_lottery_result_high',
                       'history_chose_lose', 'history_chose_earn', 'history_not_chose_lose', 'time_spent_low',
                       'time_spent_high']
    # 'prev_expected_dm_payoff_med1',
    models_dict = {
        # 'AverageOverTrainBaseLine': train_predict_simple_baseline_model('AverageOverTrainBaseLine',
        #                                                                 binary_classification=True,
        #                                                                 use_first_round=False),
        # 'AverageRishaTrainBaseline': train_predict_simple_baseline_model('AverageRishaTrainBaseline',
        #                                                                  binary_classification=True,
        #                                                                  use_first_round=False),
        # 'BasicTextSingleModel': train_valid_base_text_model_single_round_label(max_seq_len=10),
        # 'BasicTextModel': train_valid_base_text_model(),
        # 'SimpleFeaturesModelBertPrevRound':
        #     train_test_simple_features_model('SimpleFeaturesModelBertPrevRound',
        #                                      'all_data_single_round_label_prev_round_bert_embedding.pkl'),
        # 'SimpleFeaturesModelBertNoHistory':
        #     train_test_simple_features_model('SimpleFeaturesModelBertNoHistory',
        #                                      'all_data_single_round_label_bert_embedding.pkl'),
        # 'SimpleFeaturesModelBertPrevRoundAllHistory':
        #     train_test_simple_features_model('SimpleFeaturesModelBertPrevRoundAllHistory',
        #                                      'all_data_single_round_label_prev_round_all_history_bert_embedding.pkl'),
        # 'SimpleFeaturesModelBertAllHistory':
        #     train_test_simple_features_model('SimpleFeaturesModelBertAllHistory',
        #                                      'all_data_single_round_label_all_history_bert_embedding.pkl'),
        # 'SimpleFeaturesModelManualFeaturesAllHistoryTextFeaturesBackward':
        #     train_test_simple_features_model('SimpleFeaturesModelManualFeaturesAllHistoryTextFeaturesBackward',
        #                                      'all_data_single_round_label_all_history_text_features_manual_binary_'
        #                                      'features.pkl', backward_search=True),
        # 'SimpleFeaturesModelManualFeaturesAllHistoryFeaturesBackward':
        #     train_test_simple_features_model('SimpleFeaturesModelManualFeaturesAllHistoryFeaturesBackward',
        #                                      'all_data_single_round_label_all_history_manual_binary_features.pkl',
        #                                      backward_search=True),
        # 'SimpleFeaturesModelManualNoBinaryFeaturesAllHistoryFeaturesBackward':
        #     train_test_simple_features_model('SimpleFeaturesModelManualNoBinaryFeaturesAllHistoryFeaturesBackward',
        #                                      'all_data_single_round_label_all_history_manual_features.pkl',
        #                                      backward_search=True),
        # 'SimpleFeaturesModelManualFeaturesPrevRoundAllHistoryTextAndFeatures':
        #     train_test_simple_features_model('SimpleFeaturesModelManualFeaturesPrevRoundAllHistoryTextAndFeatures',
        #                                      'all_data_single_round_label_prev_round_global_alpha_0.9_all_history_text_'
        #                                      'alpha_0.8_manual_features.pkl', backward_search=False),
        # 'SimpleFeaturesModelManualFeaturesPrevRoundAllHistoryText':
        #     train_test_simple_features_model('SimpleFeaturesModelManualFeaturesPrevRoundAllHistoryText0.8Global0.9',
        #                                      'all_data_single_round_label_prev_round_all_history_alpha_0.9_'
        #                                      'all_history_text_alpha_0.8_manual_features.pkl'),
        # 'SimpleFeaturesModelManualFeaturesPrevRoundAllHistory':
        #     train_test_simple_features_model('SimpleFeaturesModelManualFeaturesPrevRoundAllHistory',
        #                                      'all_data_single_round_label_prev_round_all_history_manual_binary_'
        #                                      'features.pkl'),
        # 'SimpleFeaturesModelManualFeaturesPrevRound':
        #     train_test_simple_features_model('SimpleFeaturesModelManualFeaturesPrevRound',
        #                                      'all_data_single_round_label_prev_round_manual_binary_features.pkl'),
        # 'BasicManualFeaturesTextDecisionResultOnePreviousRoundModel':
        #     train_valid_base_text_decision_results_ep_fix_text_features_model(
        #         model_name='BasicManualFeaturesTextDecisionResultOnePreviousRoundModel',
        #         single_round_label=True, use_only_prev_round=True,
        #         train_data_file_name='train_data_1_10_single_round_label_seq_manual_binary_features.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label_manual_binary_features.pkl',
        #         numbers_columns=numbers_columns),
        # 'BasicManualFeaturesTextDecisionResultAllHistoryGlobalFeaturesModel':
        #     train_valid_base_text_decision_results_ep_fix_text_features_model(
        #         model_name='BasicManualFeaturesTextDecisionResultAllHistoryText0.8GlobalFeatures0.8Model',
        #         single_round_label=True, use_only_prev_round=False,
        #         train_data_file_name='train_data_1_10_single_round_label_seq_global_alpha_0.8_'
        #                              'all_history_text_alpha_0.8_manual_binary_features.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label_seq_global_alpha_0.8_'
        #                                   'all_history_text_alpha_0.8_manual_binary_features.pkl',
        #         numbers_columns=numbers_columns),
        # 'BasicManualFeaturesTextDecisionResultNoHistoryModel':
        #     train_valid_base_text_decision_results_ep_fix_text_features_model(
        #         model_name='BasicManualFeaturesTextDecisionResultAllHistoryModel',
        #         single_round_label=True, use_only_prev_round=False, no_history=True, func_batch_size=10,
        #         train_data_file_name='train_data_1_10_single_round_label_seq_manual_binary_features.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label_seq_manual_binary_features.pkl',
        #         numbers_columns=numbers_columns),
        # 'BasicTexDecisionResultModel':
        #     train_valid_base_text_decision_results_ep_model(model_name='BasicTexDecisionResultModel',
        #         single_round_label=True, use_only_prev_round=True,
        #         train_data_file_name='train_data_1_10_single_round_label.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label.pkl'),
        # 'BasicBERTFeaturesTextDecisionResultOnePreviousRoundModel':
        #     train_valid_base_text_decision_results_ep_fix_text_features_model(
        #         model_name='BasicBERTFeaturesTextDecisionResultOnePreviousRoundModel',
        #         single_round_label=True, use_only_prev_round=True,
        #         train_data_file_name='train_data_1_10_single_round_label_seq_bert_embedding.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label_seq_bert_embedding.pkl',
        #         numbers_columns=numbers_columns),
        # 'BasicBERTFeaturesTextDecisionResultAllHistoryPrevRoundModel':
        #     train_valid_base_text_decision_results_ep_fix_text_features_model(
        #         model_name='BasicBERTFeaturesTextDecisionResultAllHistoryPrevRoundModel',
        #         single_round_label=True, use_only_prev_round=False, func_batch_size=4,
        #         train_data_file_name=
        #         'train_data_1_10_single_round_label_seq_prev_round_global_alpha_0.9_bert_embedding.pkl',
        #         validation_data_file_name=
        #         'validation_data_1_10_single_round_label_seq_prev_round_global_alpha_0.9_bert_embedding.pkl',
        #         numbers_columns=numbers_columns),
        # 'BasicBERTFeaturesTextDecisionResultNoHistoryModel':
            # train_valid_base_text_decision_results_ep_fix_text_features_model(
            #     model_name='BasicBERTFeaturesTextDecisionResultNoHistoryModel',
            #     single_round_label=True, use_only_prev_round=False, no_history=True, func_batch_size=10,
            #     train_data_file_name=
            #     'train_data_1_10_single_round_label_seq_prev_round_global_alpha_0.9_bert_embedding.pkl',
            #     validation_data_file_name=
            #     'validation_data_1_10_single_round_label_seq_prev_round_global_alpha_0.9_bert_embedding.pkl',
            #     numbers_columns=numbers_columns),
        # 'SimpleFeaturesModelNoTextAllHistory':
        #     train_test_simple_features_model('SimpleFeaturesModelNoTextAllHistory',
        #                                      'all_data_single_round_label_all_history_no_text.pkl',
        #                                      backward_search=False),
        # 'SimpleFeaturesModelNoTextAllHistoryPrevRound':
        #     train_test_simple_features_model('SimpleFeaturesModelNoTextAllHistoryPrevRound',
        #                                      'all_data_single_round_label_prev_round_all_history_no_text.pkl',
        #                                      backward_search=False),
        # 'SimpleFeaturesModelNoTextPrevRound':
        #     train_test_simple_features_model('SimpleFeaturesModelNoTextPrevRound',
        #                                      'all_data_single_round_label_prev_round_no_text.pkl',
        #                                      backward_search=False),
        # 'SimpleFeaturesModelNoTextAllHistoryBackward':
        #     train_test_simple_features_model('SimpleFeaturesModelNoTextAllHistoryBackward',
        #                                      'all_data_single_round_label_all_history_no_text.pkl',
        #                                      backward_search=True),
        # 'SimpleFeaturesModelNoTextAllHistoryPrevRoundBackward':
        #     train_test_simple_features_model('SimpleFeaturesModelNoTextAllHistoryPrevRoundBackward',
        #                                      'all_data_single_round_label_prev_round_all_history_no_text.pkl',
        #                                      backward_search=True),
        # 'SimpleFeaturesModelNoTextPrevRoundBackward':
        #     train_test_simple_features_model('SimpleFeaturesModelNoTextPrevRoundBackward',
        #                                      'all_data_single_round_label_prev_round_no_text.pkl',
        #                                      backward_search=True),
        # 'SimpleFeaturesModelNumericConditionPrevRoundAllHistoryFeatures':
        #     train_test_simple_features_model('SimpleFeaturesModelNumericConditionPrevRoundAllHistoryFeatures',
        #                                      'all_data_single_round_label_prev_round_global_alpha_0.9_all_history_text_'
        #                                      'alpha_0.8_no_text.pkl', backward_search=False),
        # 'SimpleFeaturesModelVerbalConditionAllHistoryFeaturesAndText':
        #     train_test_simple_features_model('SimpleFeaturesModelVerbalConditionAllHistoryFeaturesAndText',
        #                                      'all_data_single_round_label_all_history_features_all_history_text_manual_'
        #                                      'features_verbal_data.pkl', backward_search=False),
        'SimpleFeaturesModelVerbalConditionAllHistoryFeaturesAndTextSingleRound':
            train_test_simple_features_model('SimpleFeaturesModelVerbalConditionAllHistoryFeaturesAndTextSingleRound',
                                             'all_data_single_round_label_prev_round_prev_round_text_global_alpha_0.9_'
                                             'all_history_text_average_with_alpha_0.8_manual_features_verbal_data.pkl',
                                             backward_search=False),
        # 'BasicModelSentimentAnalysis':
        #     train_valid_base_text_decision_fix_text_features_model(
        #         model_name='BasicModelSentimentAnalysis',
        #         single_round_label=True, use_only_prev_round=False, no_history=True, func_batch_size=10,
        #         train_data_file_name='labeledTrainData_bert_embedding.pkl',
        #         validation_data_file_name='labeledTrainData_bert_embedding.pkl',
        #         numbers_columns=[], add_numeric_data=False),
    }

    for model in models_dict.keys():
        models_dict[model]


if __name__ == '__main__':
    main()
