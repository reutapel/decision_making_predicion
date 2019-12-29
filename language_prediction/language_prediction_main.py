from language_prediction.train_test_models import *


def main():

    models_dict = {
        # 'AverageOverTrainBaseLine': train_predict_simple_baseline_model('AverageOverTrainBaseLine',
        #                                                                 binary_classification=True,
        #                                                                 use_first_round=False),
        # 'AverageRishaTrainBaseline': train_predict_simple_baseline_model('AverageRishaTrainBaseline',
        #                                                                  binary_classification=True,
        #                                                                  use_first_round=False),
        # 'BasicTextSingleModel': train_valid_base_text_model_single_round_label(max_seq_len=10),
        # 'BasicTextModel': train_valid_base_text_model(),
        'SimpleFeaturesModelPrevRound':
            train_test_simple_features_model('SimpleFeaturesModelPrevRound',
                                             'all_data_single_round_label_prev_round_bert_embedding.pkl'),
        'SimpleFeaturesModelNoHistory':
            train_test_simple_features_model('SimpleFeaturesModelNoHistory',
                                             'all_data_single_round_label_bert_embedding.pkl'),
        # 'BasicManualFeaturesTextDecisionResultOnePreviousRoundModel':
        #     train_valid_base_text_decision_results_ep_fix_text_features_model(
        #         model_name='BasicManualFeaturesTextDecisionResultOnePreviousRoundModel',
        #         single_round_label=True, use_only_prev_round=True,
        #         train_data_file_name='train_data_1_10_single_round_label_seq_manual_binary_features.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label_manual_binary_features.pkl'),
        # 'BasicManualFeaturesTextDecisionResultAllHistoryModel':
        #     train_valid_base_text_decision_results_ep_fix_text_features_model(
        #         model_name='BasicManualFeaturesTextDecisionResultAllHistoryModel',
        #         single_round_label=True, use_only_prev_round=False,
        #         train_data_file_name='train_data_1_10_single_round_label_seq_manual_binary_features.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label_manual_binary_features.pkl'),
        # 'BasicManualFeaturesTextDecisionResultNoHistoryModel':
        #     train_valid_base_text_decision_results_ep_fix_text_features_model(
        #         model_name='BasicManualFeaturesTextDecisionResultAllHistoryModel',
        #         single_round_label=True, use_only_prev_round=False, no_history=True, func_batch_size=10,
        #         train_data_file_name='train_data_1_10_single_round_label_seq_manual_binary_features.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label_seq_manual_binary_features.pkl'),
        # 'BasicTexDecisionResultModel':
        #     train_valid_base_text_decision_results_ep_model(model_name='BasicTexDecisionResultModel',
        #         single_round_label=True, use_only_prev_round=True,
        #         train_data_file_name='train_data_1_10_single_round_label.pkl',
        #         validation_data_file_name='validation_data_1_10_single_round_label.pkl'),
        'BasicBERTFeaturesTextDecisionResultOnePreviousRoundModel':
            train_valid_base_text_decision_results_ep_fix_text_features_model(
                model_name='BasicBERTFeaturesTextDecisionResultOnePreviousRoundModel',
                single_round_label=True, use_only_prev_round=True,
                train_data_file_name='train_data_1_10_single_round_label_seq_bert_embedding.pkl',
                validation_data_file_name='validation_data_1_10_single_round_label_seq_bert_embedding.pkl'),
        'BasicBERTFeaturesTextDecisionResultAllHistoryModel':
            train_valid_base_text_decision_results_ep_fix_text_features_model(
                model_name='BasicBERTFeaturesTextDecisionResultAllHistoryModel',
                single_round_label=True, use_only_prev_round=False,
                train_data_file_name='train_data_1_10_single_round_label_seq_bert_embedding.pkl',
                validation_data_file_name='validation_data_1_10_single_round_label_seq_bert_embedding.pkl'),
        'BasicBERTFeaturesTextDecisionResultNoHistoryModel':
            train_valid_base_text_decision_results_ep_fix_text_features_model(
                model_name='BasicBERTFeaturesTextDecisionResultNoHistoryModel',
                single_round_label=True, use_only_prev_round=False, no_history=True, func_batch_size=10,
                train_data_file_name='train_data_1_10_single_round_label_seq_bert_embedding.pkl',
                validation_data_file_name='validation_data_1_10_single_round_label_seq_bert_embedding.pkl'),
    }

    for model in models_dict.keys():
        models_dict[model]


if __name__ == '__main__':
    main()
