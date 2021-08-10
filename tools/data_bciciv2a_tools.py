import os
import numpy as np
import torch
from collections import OrderedDict
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.signalproc import bandpass_cnt, exponential_running_standardize
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
import logging
from braindecode.mne_ext.signalproc import mne_apply
import pickle

log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(level=logging.INFO)


def main():
    # data_dir: gdf path to load.
    # output_dir: pkl path to save.
    data_dir = "~/dataset_ubuntu/BCICIV_2a_gdf"
    output_dir = '~/dataset_ubuntu/BCICIV_2a_pkl'

    parent_output_dir = os.path.abspath(os.path.join(output_dir, os.pardir))
    assert os.path.exists(parent_output_dir), \
        "Parent directory of given output directory does not exist"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get file paths:
    train_data_paths, test_data_paths = get_paths_raw_data(data_dir)

    # Frequency filter low cut.
    low_cut_hz = 0.5

    # Process and save data.
    save_processed_dataset(train_data_paths, test_data_paths, output_dir,
                           low_cut_hz)


def get_paths_raw_data(data_dir):
    subject_ids = [x for x in range(1, 10)]

    train_data_paths = [{'gdf': data_dir + f"/A0{subject_id}T.gdf",
                         'mat': data_dir + f"/A0{subject_id}T.mat"}
                        for subject_id in subject_ids]
    test_data_paths = [{'gdf': data_dir + f"/A0{subject_id}E.gdf",
                        'mat': data_dir + f"/A0{subject_id}E.mat"}
                       for subject_id in subject_ids]

    return train_data_paths, test_data_paths


def save_processed_dataset(train_filenames, test_filenames, output_dir,
                           low_cut_hz):
    train_data, test_data = {}, {}
    for train_filename, test_filename in zip(train_filenames, test_filenames):
        subject_id = train_filename['mat'].split('/')[-1][2:3]
        log.info("Processing data...")
        full_train_set = process_bbci_data(train_filename['gdf'],
                                           train_filename['mat'], low_cut_hz)
        test_set = process_bbci_data(test_filename['gdf'],
                                     test_filename['mat'], low_cut_hz)
        train_data[subject_id] = {'X': full_train_set.X, 'y': full_train_set.y}
        test_data[subject_id] = {'X': test_set.X, 'y': test_set.y}
        log.info(f"Done processing data subject {subject_id}\n")
    log.info("Saving processed data...")
    with open(os.path.join(output_dir, 'bciciv_2a_train.pkl'), 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, 'bciciv_2a_test.pkl'), 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)


def process_bbci_data(filename, labels_filename, low_cut_hz):
    ival = [-500, 4000]
    high_cut_hz = 100
    factor_new = 1e-3
    init_block_size = 1000

    loader = BCICompetition4Set2A(filename, labels_filename=labels_filename)
    cnt = loader.load()

    # Preprocessing
    try:
        cnt = cnt.drop_channels(
            ['STI 014', 'EOG-left', 'EOG-central', 'EOG-right'])
    except ValueError:
        cnt = cnt.drop_channels(
            ['EOG-left', 'EOG-central', 'EOG-right'])
    assert len(cnt.ch_names) == 22

    # lets convert to millvolt for numerical stability of next operations
    cnt = mne_apply(lambda a: a * 1e6, cnt)
    cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz,
                               cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), cnt)

    # the experiment results show no benefit from using exponential_running_standardize for bciciv-2a dataset.
    # cnt = mne_apply(lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
    #                                                           init_block_size=init_block_size, eps=1e-4).T, cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset


def load_bciciv2a_data_single_subject(filename, subject_id, to_tensor=True):
    subject_id = str(subject_id)
    train_path = os.path.join(filename, 'bciciv_2a_train.pkl')
    test_path = os.path.join(filename, 'bciciv_2a_test.pkl')
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
    test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    main()

