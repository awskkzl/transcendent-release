import logging
import os

import torch
from termcolor import cprint
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

import transcend.calibration as calibration
import transcend.data as data
import transcend.scores as scores
import transcend.thresholding as thresholding

import pickle

import numpy as np
from tesseract import loader, spatial
from mlp.DrebinMLP import mlp_predict, load_checkpoint, DrebinMLP


def main():
    # ---------------------------------------- #
    # 0. Prelude                               #
    # ---------------------------------------- #
    logging.basicConfig(filename='ICE_DL.log', level=logging.INFO, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    thresholds = 'constrained-search'
    criteria = 'cred+conf'
    rs_max = 'f1_k:0.90'
    rs_con = 'kept_pos_perc: 0.85,kept_neg_perc: 0.85'
    rs_samples = 10000
    pval_consider = 'cal-only'
    argstest = ''
    argstrain = ''
    batch_size = 64

    dataset, t = load_dataset(full_feature=False)
    _features, _labels = dataset[0]
    n_features = _features.shape[0]
    years = np.array([dt.year for dt in t])
    months = np.array([dt.month for dt in t])
    X, y = dataset.tensors

    logging.info('Partition {} training, testing, and timestamps...'.format('MLP'))

    train_index = np.where(years == 2014)[0]
    np.random.shuffle(train_index)

    test_index = []
    for year in range(2015, 2019):
        for month in range(1, 13):
            test_index.append(np.where((years == year) & (months == month))[0])

    logging.info('Loaded: {}'.format(X.shape, y.shape))

    cal_size = 0.34
    saved_data_folder = os.path.join('models', 'ice-{}-{}'.format('10', 'deepdrebin'))
    if not os.path.exists(saved_data_folder):
        os.makedirs(saved_data_folder)

    # ---------------------------------------- #
    # 1. Calibration                           #
    # ---------------------------------------- #

    logging.info('Training calibration set...')

    proper_train_index, cal_index = train_test_split_indices(train_index, test_size=cal_size, shuffle=True)

    cal_results_dict = calibration.train_calibration_ice_mlp(
        dataset,
        batch_size=batch_size,
        train_index=proper_train_index,
        cal_index=cal_index,
        fold_index=cal_size,
        saved_data_folder=saved_data_folder
    )

    # ---------------------------------------- #
    # 2. Find Calibration Thresholds           #
    # ---------------------------------------- #

    cred_p_val_cal = cal_results_dict['cred_p_val_cal']
    conf_p_val_cal = cal_results_dict['conf_p_val_cal']

    pred_cal = cal_results_dict['pred_cal']
    groundtruth_cal = cal_results_dict['groundtruth_cal']

    if thresholds == 'random-search':
        scores_p_val_cal = package_cred_conf(
            cred_p_val_cal, conf_p_val_cal, criteria)

        p_val_found_thresholds = thresholding.find_random_search_thresholds(
            scores=scores_p_val_cal,
            predicted_labels=pred_cal,
            groundtruth_labels=groundtruth_cal,
            max_metrics=rs_max,
            max_samples=rs_samples)

    elif thresholds == 'constrained-search':

        scores_p_val_cal = package_cred_conf(
            cred_p_val_cal, conf_p_val_cal, criteria)

        # is a computationally-intensive task, so caching them helps if
        # re-running experiments. Make sure stale caches are properly
        # deleted, tho!

        # Cache pval scores in thresholding.find_random_search_thresholds_with_constraints
        statistic_name = 'mlp_scores_p_val_cal_ice_{}.p'.format(cal_size)
        statistic_name = os.path.join(saved_data_folder, statistic_name)
        # groundtruth_cal_list = groundtruth_cal.tolist()

        p_val_found_thresholds = thresholding.find_random_search_thresholds_with_constraints(
            scores=scores_p_val_cal,
            predicted_labels=pred_cal,
            groundtruth_labels=groundtruth_cal,
            maximise_vals=rs_max,
            constraint_vals=rs_con,
            # statistic_name=statistic_name,
            max_samples=rs_samples)

    else:
        msg = 'Unknown option: thresholds = {}'.format(thresholds)
        logging.critical(msg)
        raise ValueError(msg)

    # ---------------------------------------- #
    # 3. Generate 'Full' Model for Deployment  #
    # ---------------------------------------- #

    logging.info('Beginning TEST phase.')

    logging.info('Training model on full training set...')

    model_name = 'best_mlp_cal_fold_ice_{}.p'.format(cal_size)
    model_name = os.path.join(saved_data_folder, model_name)

    if os.path.exists(model_name):
        logging.warning('FOUND SAVED ICE PROPER TRAIN MODEL.')
        model = DrebinMLP(input_size=n_features)
        _, _, _ = load_checkpoint(model_name, model, None)
    else:
        raise EOFError('should already have the model')

    # ---------------------------------------- #
    # 4. Score and Predict Test Observations   #
    # ---------------------------------------- #

    logging.info('Loading {} test features...'.format(argstest))

    time_series_p_val_results, time_series_p_vals, p_val_keep_masks, \
        time_series_proba_results, time_series_probas, proba_keep_masks = [], [], [], [], [], []

    for index in test_index:
        test_sampler = SubsetRandomSampler(index)
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler)

        # Probability scores

        # logging.info('Getting probabilities for test ({})...'.format(argstest))
        # probas_test, pred_proba_test = scores.get_svm_probs(svm, X_test)

        # P-value scores

        logging.info('Computing p-values for test ({})...'.format(argstest))
        pred_test, y_test, X_test = mlp_predict(test_loader, model, model_name)

        saved_data_name = 'p_vals_ncms_{}_mlp_full_test_phase.p'.format(
            pval_consider.replace('-', '_'))
        saved_data_name = os.path.join(saved_data_folder, saved_data_name)

        if pval_consider == 'cal-only':
            logging.info('Using calibration ncms...')
            ncms = cal_results_dict['ncms_cal']
            groundtruth = groundtruth_cal
        else:
            raise ValueError('Unknown value: args.pval_consider={}'.format(pval_consider))

        logging.info('Getting NCMs for test ({})...'.format(argstest))
        ncms_full_test = scores.get_mlp_ncms(model, X_test, pred_test)

        p_val_test_dict = scores.compute_p_values_cred_and_conf(
            train_ncms=ncms,
            groundtruth_train=groundtruth,
            test_ncms=ncms_full_test,
            y_test=pred_test)
        data.cache_data(p_val_test_dict, saved_data_name)

        # ---------------------------------------- #
        # 5. Apply Thresholds, Compare Results     #
        # ---------------------------------------- #

        report_str = ''

        def print_and_extend(report_line):
            nonlocal report_str
            cprint(report_line, 'red')
            report_str += report_line + '\n'


        if thresholds in ('random-search', 'constrained-search'):

            print_and_extend('=' * 40)
            print_and_extend('[P-VALS] Threshold with random grid search')
            print_thresholds(p_val_found_thresholds)

            results, keep_mask = thresholding.test_with_rejection(
                binary_thresholds=p_val_found_thresholds,
                test_scores=p_val_test_dict,
                groundtruth_labels=y_test,
                predicted_labels=pred_test)

            time_series_p_val_results.append(results)
            time_series_p_vals.append(p_val_test_dict)
            p_val_keep_masks.append(keep_mask)

            report_str += thresholding.report_results(results)

        else:
            raise ValueError(
                'Unknown option: thresholds = {}'.format(thresholds))

        # data.save_results(report_str, args)

    data.cache_data(time_series_p_val_results, 'timeseries_cred_conf/ice_p_val_mlp_results.p')
    data.cache_data(time_series_p_vals, 'timeseries_cred_conf/ice_p_vals_mlp.p')
    data.cache_data(p_val_keep_masks, 'timeseries_cred_conf/ice_p_val_mlp_keep_masks.p')
    data.cache_data(time_series_proba_results, 'timeseries_cred_conf/ice_proba_mlp_results.p')
    data.cache_data(time_series_probas, 'timeseries_cred_conf/ice_probas_mlp.p')
    data.cache_data(proba_keep_masks, 'timeseries_cred_conf/ice_proba_mlp_keep_masks.p')


def package_cred_conf(cred_values, conf_values, criteria):
    package = {}

    if 'cred' in criteria:
        package['cred'] = cred_values
    if 'conf' in criteria:
        package['conf'] = conf_values

    return package


def load_dataset(full_feature=False):
    X, y, t, _ = loader.load_features(os.path.join('features/', 'extended-features'))
    if not full_feature:
        idx = pickle.load(open(os.path.join('features/', 'selected_feature_idx.p'), "rb"))
        X = X[:, idx]
    X = X.tocoo()
    X_tensor_tmp = torch.sparse.LongTensor(torch.LongTensor([X.row.tolist(), X.col.tolist()]),
                                           torch.LongTensor(X.data.astype(np.int32)))

    # Convert the csr_matrix to numpy array, then to tensor
    X_tensor = X_tensor_tmp.to_dense().type(torch.FloatTensor)
    y_tensor = torch.from_numpy(y.astype(np.int_))

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return dataset, t


def print_thresholds(binary_thresholds):
    # Display per-class thresholds
    if 'cred' in binary_thresholds:
        s = ('Cred thresholds: mw {:.6f}, gw {:.6f}'.format(
            binary_thresholds['cred']['mw'],
            binary_thresholds['cred']['gw']))
    if 'conf' in binary_thresholds:
        s = ('Conf thresholds: mw {:.6f}, gw {:.6f}'.format(
            binary_thresholds['conf']['mw'],
            binary_thresholds['conf']['gw']))
    logging.info(s)
    return s


def train_test_split_indices(
        indices,
        test_size=None, train_size=None,
        random_state=None, shuffle=True,
        stratify=None
):
    """Split indices into random train and test subsets.

    Parameters
    ----------
    indices : array-like
        Array of indices to split.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    train_index, test_index : list
        List containing train-test split of indices.
    """

    if not shuffle and stratify is not None:
        raise ValueError("Stratified train/test split is not implemented for shuffle=False")

    n_samples = len(indices)

    if stratify is not None:
        CVClass = StratifiedShuffleSplit
    else:
        CVClass = ShuffleSplit

    cv = CVClass(test_size=test_size, train_size=train_size, random_state=random_state)

    train_index, test_index = next(cv.split(X=np.zeros(n_samples), y=stratify))

    # Apply the split to the provided indices
    train_indices = indices[train_index]
    test_indices = indices[test_index]

    return train_indices, test_indices


if __name__ == '__main__':
    main()
