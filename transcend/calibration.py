# -*- coding: utf-8 -*-

"""
calibration.py
~~~~~~~~~~~~~~

Functions for partitioning and training proper training and calibration sets.

"""
import logging
import multiprocessing as mp
import os
from itertools import repeat

import numpy as np
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.svm import SVC
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm

import transcend.data as data
import transcend.scores as scores


def train_calibration_ice_mlp(dataset, batch_size, train_index, cal_index, fold_index, saved_data_folder='.'):
    """
    Train an MLP model for a calibration set and perform analysis for a single fold.

    The function trains an MLP model using a proper training subset, and then
    uses a calibration subset for evaluation and analysis. It computes Non-Conformity Measures (NCMs),
    predictions, and p-values for credibility and confidence, which are essential for calibration analysis.

    Args:
        dataset (TensorDataset): The dataset containing the features and labels.
        batch_size (int): The batch size for training and calibration.
        proper_train_index (list or array): Indices for the proper training set.
        cal_index (list or array): Indices for the calibration set.
        fold_index (int or float): An index to identify the current fold (used for caching).
        saved_data_folder (str, optional): Path to the folder for saving models and results. Defaults to '.'.

    Returns:
        dict: A dictionary containing the results of the calibration analysis with the following structure:
            - 'cred_p_val_cal': Calibration credibility p-values.
            - 'conf_p_val_cal': Calibration confidence p-values.
            - 'ncms_cal': Calibration NCMs.
            - 'pred_cal': Calibration predictions.
            - 'groundtruth_cal': Calibration ground truth labels.
            - 'model': The trained MLP model.
    """

    # Train model with proper training
    from mlp.DrebinMLP import DrebinMLP, mlp_train_model, mlp_predict

    model_name = 'mlp_cal_fold_ice_{}.p'.format(fold_index)
    best_model_name = os.path.join(saved_data_folder, 'best_' + model_name)
    last_model_name = os.path.join(saved_data_folder, 'last_' + model_name)
    learning_rate = 0.05
    num_epochs = 10

    tmp = int(0.7*len(train_index))

    proper_train_index = train_index[:tmp]
    val_index = train_index[tmp:]

    proper_train_sampler = SubsetRandomSampler(proper_train_index)
    val_sampler = SubsetRandomSampler(val_index)
    cal_sampler = SubsetRandomSampler(cal_index)

    proper_train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=proper_train_sampler)
    val_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=val_sampler)
    cal_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=cal_sampler)

    features, labels = dataset[0]
    model = DrebinMLP(features.shape[0])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.exists(best_model_name):
        mlp_train_model(proper_train_loader, val_loader, model, optimizer, criterion, num_epochs, best_model_name, last_model_name)

    # Get ncms for calibration fold
    logging.debug('Getting calibration ncms for fold {}...'.format(fold_index))

    pred_cal_fold, groundtruth_cal_fold, X_cal = mlp_predict(cal_loader, model, best_model_name)

    # Compute p values for calibration fold

    logging.debug('Computing cal p values for fold {}...'.format(fold_index))

    saved_ncms_name = 'ncms_mlp_cal_fold_{}.p'.format(fold_index)
    saved_ncms_name = os.path.join(saved_data_folder, saved_ncms_name)

    if os.path.exists(saved_ncms_name):
        ncms_cal_fold = data.load_cached_data(saved_ncms_name)
    else:
        ncms_cal_fold = scores.get_mlp_ncms(model, X_cal, groundtruth_cal_fold)
        data.cache_data(ncms_cal_fold, saved_ncms_name)

    saved_pvals_name = 'p_vals_mlp_cal_fold_{}.p'.format(fold_index)
    saved_pvals_name = os.path.join(saved_data_folder, saved_pvals_name)

    if os.path.exists(saved_pvals_name):
        p_val_cal_fold_dict = data.load_cached_data(saved_pvals_name)
    else:
        p_val_cal_fold_dict = scores.compute_p_values_cred_and_conf(
            train_ncms=ncms_cal_fold,
            groundtruth_train=groundtruth_cal_fold,
            test_ncms=ncms_cal_fold,
            y_test=groundtruth_cal_fold)
        data.cache_data(p_val_cal_fold_dict, saved_pvals_name)

    # Compute values for calibration probabilities
    # logging.debug('Computing cal probas for fold {}...'.format(fold_index))
    # probas_cal_fold, pred_proba_cal_fold = scores.get_svm_probs(svm, X_cal)

    return {
        # Calibration credibility p values
        'cred_p_val_cal': p_val_cal_fold_dict['cred'],
        # Calibration confidence p values
        'conf_p_val_cal': p_val_cal_fold_dict['conf'],
        'ncms_cal': ncms_cal_fold,  # Calibration NCMs
        'pred_cal': pred_cal_fold,  # Calibration predictions
        'groundtruth_cal': groundtruth_cal_fold,  # Calibration groundtruth
        # 'probas_cal': probas_cal_fold,  # Calibration probabilities
        # 'pred_proba_cal': pred_proba_cal_fold,  # Calibration predictions
        'model': model
    }


def train_calibration_set(
        X_train, y_train, n_folds=10, ncpu=1, saved_data_folder='.', loo=False):
    """Train the calibration set in order to find thresholds.

    In order to reduce overfitting, calibration training is done by
    partitioning the calibration set into multiple folds and training
    them separately.

    Args:
        X_train (np.ndarray): Set of calibration training features.
        y_train (np.ndarray): Set of calibration training ground truths.
        n_folds (int): The number of partitions to use.
        ncpu (int): The number of processes to use.
        saved_data_folder (str): The folder in which to cache all of the
            trained models and computed p-values.
        loo (bool): Flag for leave-one-out splits. Default is false. 

    See Also:
        - `train_calibration_fold`

    Returns:
        list: A list of result dictionaries containing information generated
            for each training fold.

    """
    if loo:
        skf = LeaveOneOut()
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=21)

    folds = skf.split(X_train, y_train)

    fold_generator = ({
        'X_proper_train_fold': X_train[proper_train_index, :],
        'y_proper_train_fold': y_train[proper_train_index],
        'X_cal_fold': X_train[cal_index, :],
        'y_cal_fold': y_train[cal_index],
        'fold_index': idx,
        'folder': saved_data_folder
    } for idx, (proper_train_index, cal_index) in enumerate(folds))

    results_list = []

    with mp.Pool(processes=ncpu) as pool:
        n_splits = skf.get_n_splits(X_train, y_train)

        for res in tqdm(pool.imap(train_calibration_fold, fold_generator), total=n_splits):
            results_list.append(res)

    return list(results_list)


def concatenate_calibration_set_results(results_list):
    """Concatenate all the fold results from `train_calibration_set`.

    Args:
        results_list: List of dictionaries of results.

    See Also:
        - `train_calibration_set`
        - `train_calibration_fold`

    Returns:
        dict: A dictionary of lists containing concatenated results.

    """
    results_dict = {}

    for result in results_list:
        for key, value in result.items():
            key = key.replace('_fold', '')
            if key not in results_dict:
                results_dict[key] = np.array(value)
            else:
                concatenated = np.concatenate(
                    (results_dict[key], np.array(value)))
                results_dict[key] = concatenated

    return results_dict


def train_calibration_ice(
        X_proper_train, X_cal,
        y_proper_train, y_cal, fold_index, saved_data_folder='.'):
    """Train calibration set (for a single fold).

    Quite a bit of information is needed here for the later p-value
    computation and probability comparison. The returned dictionary has
    the following structure:

        'cred_p_val_cal_fold'  -->  # Calibration credibility p values
        'conf_p_val_cal_fold'  -->  # Calibration confidence p values
        'ncms_cal_fold'        -->  # Calibration NCMs
        'pred_cal_fold'        -->  # Calibration predictions
        'groundtruth_cal_fold' -->  # Calibration groundtruth
        'probas_cal_fold'      -->  # Calibration probabilities
        'pred_proba_cal_fold'  -->  # Calibration predictions

    Args:
        X_proper_train (np.ndarray): Features for the 'proper training
            set' partition.
        X_cal (np.ndarray): Features for a single calibration set
            partition.
        y_proper_train (np.ndarray): Ground truths for the 'proper
            training set' partition.
        y_cal (np.ndarray): Ground truths for a single calibration set
            partition.
        fold_index: An index to identify the current fold (used for caching).

    Returns:
        dict: Fold results, structure as in the docstring above.

    """
    # Train model with proper training

    model_name = 'svm_cal_fold_{}.p'.format(fold_index)
    model_name = os.path.join(saved_data_folder, model_name)

    if os.path.exists(model_name):
        svm = data.load_cached_data(model_name)
    else:
        svm = SVC(probability=True, kernel='linear', verbose=True)
        svm.fit(X_proper_train, y_proper_train)
        data.cache_data(svm, model_name)

    # Get ncms for proper training fold

    logging.debug('Getting training ncms for fold {}...'.format(fold_index))
    groundtruth_proper_train_fold = y_proper_train

    # Get ncms for calibration fold

    logging.debug('Getting calibration ncms for fold {}...'.format(fold_index))
    pred_cal_fold = svm.predict(X_cal)
    groundtruth_cal_fold = y_cal

    # Compute p values for calibration fold

    logging.debug('Computing cal p values for fold {}...'.format(fold_index))

    saved_ncms_name = 'ncms_svm_cal_fold_{}.p'.format(fold_index)
    saved_ncms_name = os.path.join(saved_data_folder, saved_ncms_name)

    if os.path.exists(saved_ncms_name):
        ncms_cal_fold = data.load_cached_data(saved_ncms_name)
    else:
        ncms_cal_fold = scores.get_svm_ncms(svm, X_cal, y_cal)
        data.cache_data(ncms_cal_fold, saved_ncms_name)

    saved_pvals_name = 'p_vals_svm_cal_fold_{}.p'.format(fold_index)
    saved_pvals_name = os.path.join(saved_data_folder, saved_pvals_name)

    if os.path.exists(saved_pvals_name):
        p_val_cal_fold_dict = data.load_cached_data(saved_pvals_name)
    else:
        # TODO | Doublecheck implications of duplicating the reference
        # TODO | point in the 'train_ncms'
        p_val_cal_fold_dict = scores.compute_p_values_cred_and_conf(
            train_ncms=ncms_cal_fold,
            groundtruth_train=groundtruth_cal_fold,
            test_ncms=ncms_cal_fold,
            y_test=groundtruth_cal_fold)
        data.cache_data(p_val_cal_fold_dict, saved_pvals_name)

    # Compute values for calibration probabilities
    logging.debug('Computing cal probas for fold {}...'.format(fold_index))
    probas_cal_fold, pred_proba_cal_fold = scores.get_svm_probs(svm, X_cal)

    return {
        # Calibration credibility p values
        'cred_p_val_cal': p_val_cal_fold_dict['cred'],
        # Calibration confidence p values
        'conf_p_val_cal': p_val_cal_fold_dict['conf'],
        'ncms_cal': ncms_cal_fold,  # Calibration NCMs
        'pred_cal': pred_cal_fold,  # Calibration predictions
        'groundtruth_cal': groundtruth_cal_fold,  # Calibration groundtruth
        'probas_cal': probas_cal_fold,  # Calibration probabilities
        'pred_proba_cal': pred_proba_cal_fold,  # Calibration predictions
        'model': svm
    }


def train_calibration_fold(params):
    """Train calibration set (for a single fold).

    Quite a bit of information is needed here for the later p-value
    computation and probability comparison. The returned dictionary has
    the following structure:

        'cred_p_val_cal_fold'  -->  # Calibration credibility p values
        'conf_p_val_cal_fold'  -->  # Calibration confidence p values
        'ncms_cal_fold'        -->  # Calibration NCMs
        'pred_cal_fold'        -->  # Calibration predictions
        'groundtruth_cal_fold' -->  # Calibration groundtruth
        'probas_cal_fold'      -->  # Calibration probabilities
        'pred_proba_cal_fold'  -->  # Calibration predictions

    Args:
        params(dict):
            X_proper_train_fold (np.ndarray): Features for the 'proper training
                set' partition.
            X_cal_fold (np.ndarray): Features for a single calibration set
                partition.
            y_proper_train_fold (np.ndarray): Ground truths for the 'proper
                training set' partition.
            y_cal_fold (np.ndarray): Ground truths for a single calibration set
                partition.
            fold_index (int | str): An index to identify the current fold (used for caching).
            folder (str): The directory to save the data to.
    Returns:
        dict: Fold results, structure as in the docstring above.

    """
    # Train model with proper training

    X_proper_train_fold = params['X_proper_train_fold']
    y_proper_train_fold = params['y_proper_train_fold']
    X_cal_fold = params['X_cal_fold']
    y_cal_fold = params['y_cal_fold']
    fold_index = params['fold_index']
    saved_data_folder = params['folder']

    model_name = 'svm_cal_fold_{}.p'.format(fold_index)
    model_name = os.path.join(saved_data_folder, model_name)

    if os.path.exists(model_name):
        svm = data.load_cached_data(model_name)
    else:
        svm = SVC(probability=True, kernel='linear', verbose=True)
        svm.fit(X_proper_train_fold, y_proper_train_fold)
        data.cache_data(svm, model_name)

    # Get ncms for proper training fold

    logging.debug('Getting training ncms for fold {}...'.format(fold_index))
    groundtruth_proper_train_fold = y_proper_train_fold

    # Get ncms for calibration fold

    logging.debug('Getting calibration ncms for fold {}...'.format(fold_index))
    pred_cal_fold = svm.predict(X_cal_fold)
    groundtruth_cal_fold = y_cal_fold

    # Compute p values for calibration fold

    logging.debug('Computing cal p values for fold {}...'.format(fold_index))

    saved_ncms_name = 'ncms_svm_cal_fold_{}.p'.format(fold_index)
    saved_ncms_name = os.path.join(saved_data_folder, saved_ncms_name)

    if os.path.exists(saved_ncms_name):
        ncms_proper_train_fold, ncms_cal_fold = data.load_cached_data(
            saved_ncms_name)
    else:
        ncms_proper_train_fold = scores.get_svm_ncms(svm, X_proper_train_fold,
                                                     y_proper_train_fold)
        ncms_cal_fold = scores.get_svm_ncms(svm, X_cal_fold, y_cal_fold)
        data.cache_data((ncms_proper_train_fold, ncms_cal_fold),
                        saved_ncms_name)

    saved_pvals_name = 'p_vals_svm_cal_fold_{}.p'.format(fold_index)
    saved_pvals_name = os.path.join(saved_data_folder, saved_pvals_name)

    if os.path.exists(saved_pvals_name):
        p_val_cal_fold_dict = data.load_cached_data(saved_pvals_name)
    else:
        p_val_cal_fold_dict = scores.compute_p_values_cred_and_conf(
            train_ncms=ncms_proper_train_fold,
            groundtruth_train=groundtruth_proper_train_fold,
            test_ncms=ncms_cal_fold,
            y_test=groundtruth_cal_fold)
        data.cache_data(p_val_cal_fold_dict, saved_pvals_name)

    # Compute values for calibration probabilities
    logging.debug('Computing cal probas for fold {}...'.format(fold_index))
    probas_cal_fold, pred_proba_cal_fold = scores.get_svm_probs(svm, X_cal_fold)

    return {
        # Calibration credibility p values
        'cred_p_val_cal_fold': p_val_cal_fold_dict['cred'],
        # Calibration confidence p values
        'conf_p_val_cal_fold': p_val_cal_fold_dict['conf'],
        'ncms_cal_fold': ncms_cal_fold,  # Calibration NCMs
        'pred_cal_fold': pred_cal_fold,  # Calibration predictions
        'groundtruth_cal_fold': groundtruth_cal_fold,  # Calibration groundtruth
        'probas_cal_fold': probas_cal_fold,  # Calibration probabilities
        'pred_proba_cal_fold': pred_proba_cal_fold  # Calibration predictions
    }
