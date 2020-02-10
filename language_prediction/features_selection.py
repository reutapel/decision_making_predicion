import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from xgboost import XGBClassifier
import joblib
from numpy.ctypeslib import ndpointer
import ctypes as c
import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'data', 'verbal')


class CMIM(BaseEstimator, TransformerMixin):
    """ Class for CMIM features selection
        based on C implementation in Feature Selection Tools box
        https://github.com/Craigacp/FEAST

        Parameters
        ----------
        num_of_features : integer, default = 20

            the amount of features to choose.

        Attributes
        ----------
        feature_names : list of strings
            Names of the selected features.
            * will be available only in case data for fit is pandas DataFrame
        feature_scores : list of floats
            Scores of selected feature. The score is calculated by min{I(Xn ; Y | Xk)}
        feature_indexes : list of floats
            Indexes of selected feature.

        Examples
        --------
        # >>> import pandas as pd
        # >>> from anyml.feature_selection.filter_methods import CMIM
        # >>> from sklearn.datasets import load_boston
        # >>> from anyml.data_preprocessing.data_discretization import equal_interval_discretization
        # >>> boston = load_boston()
        # >>> X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
        # >>> y = pd.DataFrame(data=boston.target, columns=['Labels'])
        # >>> X_discrete = equal_interval_discretization(X=X, n_bins=6)
        # >>> y_discrete = equal_interval_discretization(X=y, n_bins=6)
        # >>> X_discrete = X_discrete - 1
        # >>> y_discrete = y_discrete - 1
        # >>> feast_cmim = CMIM(num_of_features=2)
        # >>> feast_cmim.fit(X_discrete, y_discrete)
        # >>> if feast_cmim.feature_names is not None:
        # >>>     selected_features = feast_cmim.feature_names
        # >>> else:
        # >>>     feature_names = X.columns.tolist()
        # >>>     selected_features = [feature_names[idx] for idx in feast_cmim.feature_indexes]
        # >>> print(selected_features)
        ['LSTAT', 'RM']
        """

    def __init__(self, num_of_features=20):
        if sys.platform.startswith('linux'):
            # in case of Linux OS load MIToolbox and then FEAST linux library (.so files)
            # cwd = os.getcwd()
            # os.chdir('{}/{}'.format(os.path.dirname(__file__), 'lib'))
            self.libFSToolbox = c.CDLL('{}/{}/{}'.format(os.path.dirname(__file__), 'lib', 'libFSToolbox.so'))
            # os.chdir(cwd)
        elif sys.platform.startswith('win'):
            # in case of Windows OS load feast windows library (.dll file)
            self.libFSToolbox = c.WinDLL('{}/{}/{}'.format(os.path.dirname(__file__), 'lib', 'libFSToolbox.dll'))
        else:
            # OSs other than Linux and Windows are not supported
            raise NotImplementedError('FeastCMIM is not supported on {} operating system'.format(sys.platform))
        self.num_of_features = num_of_features
        self.feature_indexes = None
        self.feature_scores = None
        self.feature_names = None

    def fit(self, data, labels, sample_weight=None):
        """
        Fits a defined CMIM filter to a given data set.
        data and labels are expected to be discretized

        Parameters
        ----------
        data : pandas or numpy data object
            The training input samples.
        labels : pandas or numpy data object. labels should contain a binary label.
            The label of the training samples.
        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples (1. for unweighted).
        """
        # in case the data is a pandas data frame we can store features by name
        data_labels = None
        if type(data) is pd.DataFrame:
            data_labels = data.columns.tolist()
        if sample_weight is None:
            sample_weight = np.ones(shape=labels.shape)
        if sample_weight.min() < 0 or sample_weight.sum() == 0:
            raise Exception("Invalid weights input in CMIM")
        sample_weight /= sample_weight.mean()
        if (sample_weight.shape[0] != labels.shape[0]) or (data.shape[0] != labels.shape[0]):
            raise Exception("Mismatch in CMIM input dimensions")
        # python variables adaptation for C parameters initialization
        data = np.array(data, dtype=np.uint32, order="F")
        labels = np.array(labels, dtype=np.uint32)
        sample_weight = np.array(sample_weight)
        n_samples, n_features = data.shape
        output = np.zeros(self.num_of_features).astype(np.uint)
        selected_features_score = np.zeros(self.num_of_features)
        # cast as C types
        _uintpp = ndpointer(dtype=np.uintp, ndim=1, flags='F')
        c_k = c.c_uint(self.num_of_features)
        c_no_of_samples = c.c_uint(n_samples)
        c_no_of_features = c.c_uint(n_features)
        c_feature_matrix = (data.__array_interface__['data'][0] + np.arange(data.shape[1]) * (data.strides[1])).astype(
            np.uintp)
        c_class_column = labels.ctypes.data_as(c.POINTER(c.c_uint))
        c_output_features = output.ctypes.data_as(c.POINTER(c.c_uint))
        c_weight_vector = sample_weight.ctypes.data_as(c.POINTER(c.c_double))
        c_feature_scores = selected_features_score.ctypes.data_as(c.POINTER(c.c_double))
        self.libFSToolbox.weightedCMIM.argtypes = [c.c_uint, c.c_uint, c.c_uint, _uintpp, c.POINTER(c.c_uint),
                                                   c.POINTER(c.c_double), c.POINTER(c.c_uint), c.POINTER(c.c_double)]
        self.libFSToolbox.weightedCMIM.restype = c.POINTER(c.c_uint)
        # call the C implementation
        c_selected_features = self.libFSToolbox.weightedCMIM(c_k, c_no_of_samples, c_no_of_features, c_feature_matrix,
                                                             c_class_column, c_weight_vector, c_output_features,
                                                             c_feature_scores)
        # result transition from C to Python
        features_iterator = np.fromiter(c_selected_features, dtype=np.uint, count=self.num_of_features)
        selected_features = []
        for c_selected_feature_index in features_iterator:
            selected_features.append(c_selected_feature_index)
        # store the selection results
        self.feature_scores = [c_feature_scores[idx] for idx in range(self.num_of_features)]
        self.feature_indexes = selected_features
        if data_labels is not None:
            self.feature_names = [data_labels[idx] for idx in self.feature_indexes]

        return self

    def transform(self, x):
        if isinstance(x, np.ndarray):
            return x[:, self.feature_indexes]
        elif isinstance(x, pd.DataFrame):
            return x.loc[:, self.feature_names]
        else:
            raise AttributeError('Transform accepts data of type numpy.ndarray or pandas.DataFrame')


def cor_selector(x, y, num_features):
    cor_list = []
    feature_name = x.columns.tolist()
    # calculate the correlation with y for each feature
    for i in x.columns.tolist():
        cor = np.corrcoef(x[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = x.iloc[:, np.argsort(np.abs(cor_list))[-num_features:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]

    return cor_support, cor_feature


def chi_square(x, x_norm, y, num_features):
    chi_selector = SelectKBest(chi2, k=num_features)
    chi_selector.fit(x_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = x.loc[:, chi_support].columns.tolist()

    return chi_support, chi_feature


def rfe(x, x_norm, y, num_features):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_features, step=10, verbose=5)
    rfe_selector.fit(x_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = x.loc[:, rfe_support].columns.tolist()

    return rfe_support, rfe_feature


def lasso(x, x_norm, y, num_features):
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=num_features)
    embeded_lr_selector.fit(x_norm, y)

    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = x.loc[:, embeded_lr_support].columns.tolist()

    return embeded_lr_support, embeded_lr_feature


def tree_based(x, y, num_features, model):
    embeded_rf_selector = SelectFromModel(model, max_features=num_features)
    embeded_rf_selector.fit(x, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = x.loc[:, embeded_rf_support].columns.tolist()

    return embeded_rf_support, embeded_rf_feature


def cmim(x, y, disc_bin=5, num_features=30):
    disc = KBinsDiscretizer(n_bins=disc_bin, encode='ordinal', strategy='uniform')
    x_disc = disc.fit_transform(X=x)
    cmim = CMIM(num_of_features=num_features)
    cmim.fit(x_disc, y)
    top_features = [x.columns.values[idx] for idx in cmim.feature_indexes]
    cmim_support = [True if idx in cmim.feature_indexes else False for idx in range(num_features)]

    return cmim_support, top_features


def main():
    for data_name in ['all_data_single_round_label_manual_binary_features_verbal_data',
                      # 'all_data_single_round_label_global_alpha_0.9_all_history_text_average_with_alpha_0.8_'
                      # 'manual_binary_features_verbal_data',
                      ]:
        data = joblib.load(os.path.join(data_directory, f'{data_name}.pkl'))
        y = data['label']
        x = data.copy()
        x = x.drop(['label', 'k_size', 'pair_id', 'sample_id'], axis=1)
        x_norm = MinMaxScaler().fit_transform(x)
        feature_name = x.columns.tolist()
        num_features = int(round(len(feature_name)*0.8))

        cmim_support, cmim_features = cmim(x, y, num_features=num_features)
        cor_support, cor_features = cor_selector(x, y, num_features)
        chi_support, chi_features = chi_square(x, x_norm, y, num_features)
        rfe_support, rfe_features = rfe(x, x_norm, y, num_features)
        embeded_lr_support, embeded_lr_features = lasso(x, x_norm, y, num_features)
        embeded_rf_support, embeded_rf_features =\
            tree_based(x, y, num_features, model=RandomForestClassifier(n_estimators=100))
        embeded_lgb_support, embeded_lgb_features = tree_based(x, y, num_features, model=XGBClassifier())

        # put all selection together
        feature_selection_df = pd.DataFrame(
            {'Feature': feature_name, 'Pearson': cor_support, 'Chi-2': chi_support, 'RFE': rfe_support,
             'Logistics': embeded_lr_support, 'CMIM': cmim_support,
             'Random Forest': embeded_rf_support, 'LightGBM': embeded_lgb_support})
        # count the selected times for each feature
        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
        # display the top 100
        feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
        feature_selection_df.index = range(1, len(feature_selection_df) + 1)

        feature_selection_df.to_excel(f'feature_selection_for_{data_name}.xlsx')


if __name__ == '__main__':
    main()





