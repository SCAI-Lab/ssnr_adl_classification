import os
import pandas as pd
import pickle
import yaml
from StatisticalFeatures import StatisticalFeatures
from SpectralFeatures import SpectralFeatures
from TimeFrequencyFeatures import TimeFrequencyFeatures
from EcgFeatures import EcgFeatures
import concurrent.futures
import warnings


"""
   Parameters
"""
# Muting warnings
warnings.filterwarnings("ignore")

# Load parameters from the yaml file
with open('parameters.yaml', 'r') as f:
    params = yaml.safe_load(f)
seed_number = params['seed_number']
fs = params['upsample_freq']
window_size = params['window_size']
overlap = params['overlap']
f_bands = params['f_bands']

# Converting window size and overlap to number of samples
window_size = window_size * fs
overlap = overlap * window_size
step_size = window_size - overlap

# Get the current directory
current_dir = os.getcwd()
# The directory for the intermediate data
intermediate_data_dir = os.path.join(current_dir, 'data', 'intermediate')
# Loading the segmented data
file_name = os.path.join(intermediate_data_dir, 'segmented_data.pickle')
with open(file_name, 'rb') as f:
    segmented_data = pickle.load(f)

# Defining feature generation objects
stat_feat_extractor = StatisticalFeatures(window_size=window_size)
freq_feat_extractor = SpectralFeatures(fs=fs, f_bands=f_bands)
time_freq_feat_extractor = TimeFrequencyFeatures(window_size=window_size)
ecg_feat_extractor = EcgFeatures(fs=fs)

# Defining a df_features with headers for devices, features, and labels
subject = list(segmented_data.keys())[0]
activity = list(segmented_data[subject].keys())[0]
devices = sorted(list(segmented_data[subject][activity].keys()))
# A dict for storing the 2 layer headers
headers = {}
for device in devices:
    # A list for storing feature names
    feats_names = []
    df_signals = segmented_data[subject][activity][device]
    for column in sorted(df_signals.columns):
        signal = df_signals.iloc[0:window_size][column].values
        # signal_der1 = np.gradient(signal)
        # signal_der2 = np.gradient(signal_der1)
        signal_name = f"{device}_{column}"
        if column != 'ecg':
            # Extracting the statistical features per windowed signal and its derivatives
            _, stat_feats_names = stat_feat_extractor.calculate_statistical_features(signal, signal_name)
            feats_names.extend(stat_feats_names)
            # _, stat_feats_der1_names = stat_feat_extractor.calculate_statistical_features(signal_der1, signal_name + "_der1")
            # feats_names.extend(stat_feats_der1_names)
            # _, stat_feats_der2_names = stat_feat_extractor.calculate_statistical_features(signal_der2, signal_name + "_der2")
            # feats_names.extend(stat_feats_der2_names)
            # Extracting the spectral features per windowed signal
            _, freq_feats_names = freq_feat_extractor.calculate_frequency_features(signal, signal_name)
            feats_names.extend(freq_feats_names)
            # Extracting time-freq features per windowed signal
            _, time_freq_feats_names = time_freq_feat_extractor.calculate_time_frequency_features(signal, signal_name)
            feats_names.extend(time_freq_feats_names)
        if column == 'ecg':
            # Extracting the HR-specific features per windowed signal
            _, ecg_feats_names = ecg_feat_extractor.calculate_ecg_features(signal, signal_name)
            feats_names.extend(ecg_feats_names)
    headers[device] = feats_names
headers['label'] = ['label']
df_columns = pd.MultiIndex.from_tuples([(device, feat) for device, feats in headers.items() for feat in feats])


def extract_features(subject_activity):
    print(subject_activity)
    subject, activity = subject_activity
    df_features = pd.DataFrame(columns=df_columns)
    df_len = len(segmented_data[subject][activity][devices[0]])
    num_windows = int((df_len - overlap) // step_size)
    for i_window in range(num_windows):
        start_idx = int(i_window * step_size)
        end_idx = int(start_idx + window_size)
        # With the initial corsano orientation
        # Storing features and label
        values = []
        for device in devices:
            df_signals = segmented_data[subject][activity][device]
            for column in sorted(df_signals.columns):
                signal = df_signals.iloc[start_idx:end_idx][column].values
                # signal_der1 = np.gradient(signal)
                # signal_der2 = np.gradient(signal_der1)
                signal_name = f"{device}_{column}"
                if column != 'ecg':
                    stat_feats, _ = stat_feat_extractor.calculate_statistical_features(signal, signal_name)
                    values.extend(stat_feats)
                    # stat_feats_der1, _ = stat_feat_extractor.calculate_statistical_features(signal_der1, signal_name + "_der1")
                    # values.extend(stat_feats_der1)
                    # stat_feats_der2, _ = stat_feat_extractor.calculate_statistical_features(signal_der2, signal_name + "_der2")
                    # values.extend(stat_feats_der2)
                    freq_feats, freq_feats_names = freq_feat_extractor.calculate_frequency_features(signal, signal_name)
                    values.extend(freq_feats)
                    time_freq_feats, time_freq_feats_names = time_freq_feat_extractor.calculate_time_frequency_features(signal, signal_name)
                    values.extend(time_freq_feats)
                if column == 'ecg':
                    ecg_feats, ecg_feats_names = ecg_feat_extractor.calculate_ecg_features(signal, signal_name)
                    values.extend(ecg_feats)
        values.extend([int(activity)])
        df_features.loc[len(df_features)] = values

        # With the rotated corsano orientation
        # Storing features and label
        values = []
        for device in sorted(devices):
            df_signals = segmented_data[subject][activity][device]
            for column in sorted(df_signals.columns):
                signal = df_signals.iloc[start_idx:end_idx][column].values
                # signal_der1 = np.gradient(signal)
                # signal_der2 = np.gradient(signal_der1)
                signal_name = f"{device}_{column}"
                if signal_name in ["corsano_wrist_acc_x", "corsano_wrist_acc_y"]:
                    signal = -signal
                if column != 'ecg':
                    stat_feats, _ = stat_feat_extractor.calculate_statistical_features(signal, signal_name)
                    values.extend(stat_feats)
                    # stat_feats_der1, _ = stat_feat_extractor.calculate_statistical_features(signal_der1, signal_name + "_der1")
                    # values.extend(stat_feats_der1)
                    # stat_feats_der2, _ = stat_feat_extractor.calculate_statistical_features(signal_der2, signal_name + "_der2")
                    # values.extend(stat_feats_der2)
                    freq_feats, _ = freq_feat_extractor.calculate_frequency_features(signal, signal_name)
                    values.extend(freq_feats)
                    time_freq_feats, _ = time_freq_feat_extractor.calculate_time_frequency_features(
                        signal, signal_name)
                    values.extend(time_freq_feats)
                if column == 'ecg':
                    ecg_feats, _ = ecg_feat_extractor.calculate_ecg_features(signal, signal_name)
                    values.extend(ecg_feats)
        values.extend([int(activity)])
        df_features.loc[len(df_features)] = values
    return subject, activity, df_features


if __name__ == '__main__':
    features = {}
    subject_activity_pairs = [(subject, activity) for subject in segmented_data.keys() for activity in segmented_data[subject].keys()]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(extract_features, subject_activity_pairs))

    for subject, activity, df_features in results:
        if subject not in features:
            features[subject] = []
        features[subject].append(df_features)
    for subject in features.keys():
        features[subject] = pd.concat(features[subject], axis=0, ignore_index=True)

    # Save features for last subject
    file_name = os.path.join(intermediate_data_dir, f'features_{int(window_size // fs)}.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(features, f)


