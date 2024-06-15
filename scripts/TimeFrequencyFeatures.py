import numpy as np
from .StatisticalFeatures import StatisticalFeatures
import pywt


class TimeFrequencyFeatures:
    def __init__(self,
                 window_size,
                 wavelet='db4',
                 decomposition_level=None
                 ):

        self.window_size = window_size
        self.wavelet = wavelet
        if decomposition_level is None:
            wavelet_length = len(pywt.Wavelet(wavelet).dec_lo)
            self.decomposition_level = int(np.round(np.log2(self.window_size/wavelet_length) - 1))
        else:
            self.decomposition_level = decomposition_level

        self.statistical_feature_extractor = StatisticalFeatures(self.window_size)

    def calculate_time_frequency_features(self, signal, signal_name):
        # A list for storing all the features
        feats = []
        # A list for storing feature names
        feats_names = []

        # Performing Teager-Kaiser energy operator
        signal_tkeo = self.teager_kaiser_energy_operator(signal)
        # Perform wavelet decomposition
        wavelet_coefficients = pywt.wavedec(signal, self.wavelet, level=self.decomposition_level)

        # Mean of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_mean(signal_tkeo))
        feats_names.append(f"{signal_name}_mean_of_tkeo")
        # Standard deviation of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_std(signal_tkeo))
        feats_names.append(f"{signal_name}_standard_deviation_of_tkeo")
        # Skewness of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_skewness(signal_tkeo))
        feats_names.append(f"{signal_name}_skewness_of_tkeo")
        # Kurtosis of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_kurtosis(signal_tkeo))
        feats_names.append(f"{signal_name}_kurtosis_of_tkeo")
        # Max of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_max(signal_tkeo))
        feats_names.append(f"{signal_name}_max_of_tkeo")
        # Variance of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_variance(signal_tkeo))
        feats_names.append(f"{signal_name}_variance_of_tkeo")
        # Coefficient of variation of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_coefficient_of_variation(signal_tkeo))
        feats_names.append(f"{signal_name}_coefficient_of_variation_of_tkeo")
        # Root-mean-square of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_root_mean_square(signal_tkeo))
        feats_names.append(f"{signal_name}_rms_of_tkeo")
        # Magnitude area of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_signal_magnitude_area(signal_tkeo))
        feats_names.append(f"{signal_name}_magnitude_area_of_tkeo")
        # Number of zero crossings of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_zero_crossings(signal_tkeo))
        feats_names.append(f"{signal_name}_no._of_zero_crossings_of_tkeo")
        # Number of mean crossings of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_mean_crossing(signal_tkeo))
        feats_names.append(f"{signal_name}_no._of_mean_crossings_of_tkeo")
        # Number of slope sign changes of the signal's Teager-Kaiser energy operator
        feats.extend(self.statistical_feature_extractor.calculate_slope_sign_change(signal_tkeo))
        feats_names.append(f"{signal_name}_no._of_slope_sign_changes_of_tkeo")

        for i_level in range(self.decomposition_level + 1):
            # Mean of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_mean(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_mean_of_wav_coeffs_lvl_{i_level}")
            # Standard deviation of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_std(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_std_of_wav_coeffs_{i_level}")
            # Skewness of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_skewness(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_skewness_of_wav_coeffs_{i_level}")
            # Kurtosis of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_kurtosis(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_kurtosis_of_wav_coeffs_{i_level}")
            # Median of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_median(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_median_of_wav_coeffs_lvl_{i_level}")
            # Max of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_max(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_max_of_wav_coeffs_lvl_{i_level}")
            # Min of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_min(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_min_of_wav_coeffs_lvl_{i_level}")
            # Variance of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_variance(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_var_of_wav_coeffs_lvl_{i_level}")
            # Coefficient of variation of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_coefficient_of_variation(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_coefficient_of_variation_of_wav_coeffs_lvl_{i_level}")
            # Mean absolute deviation of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_mean_absolute_deviation(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_mean_absolute_deviation_of_wav_coeffs_lvl_{i_level}")
            # Root-mean-square of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_root_mean_square(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_rms_of_wav_coeffs_lvl_{i_level}")
            # Mean absolute deviation of the signal's wavelet decomposition coefficients
            feats.extend(self.statistical_feature_extractor.calculate_interquartile_range(wavelet_coefficients[i_level]))
            feats_names.append(f"{signal_name}_iqr_of_wav_coeffs_lvl_{i_level}")

        return np.array(feats), feats_names

    def teager_kaiser_energy_operator(self, signal):
        # Calculate the TKEO
        tkeo = np.roll(signal, -1) * np.roll(signal, 1) - signal ** 2
        # The first and last elements are not valid due to roll operation
        tkeo[0] = 0
        tkeo[-1] = 0
        return tkeo
