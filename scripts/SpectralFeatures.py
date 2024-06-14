import numpy as np
from scipy.signal import welch
from scipy.integrate import simps


class SpectralFeatures:
    def __init__(self,
                 fs,
                 f_bands,
                 n_dom_freqs=5,
                 cumulative_power_thresholds=None
                 ):
        self.fs = fs
        self.f_bands = f_bands
        self.n_dom_freqs = n_dom_freqs
        if cumulative_power_thresholds is None:
            self.cumulative_power_thresholds = np.array([.5, .75, .85, .9, 0.95])
        else:
            self.cumulative_power_thresholds = cumulative_power_thresholds

    def calculate_frequency_features(self, signal, signal_name):
        # An array for storing the spectral features.
        feats = []
        # A list for storing feature names
        feats_names = []

        length = len(signal)
        hamming_window = np.hanning(length)
        modified_signal = signal * hamming_window

        # FFT (only positive frequencies)
        spectrum = np.fft.rfft(modified_signal)  # Spectrum of positive frequencies
        spectrum_magnitudes = np.abs(spectrum)  # Magnitude of positive frequencies
        spectrum_magnitudes_normalized = spectrum_magnitudes / np.sum(spectrum_magnitudes)
        
        freqs_spectrum = np.abs(np.fft.fftfreq(length, 1.0 / self.fs)[:length // 2 + 1])

        # Calculating the power spectral density using Welch's method.
        freqs_psd, psd = welch(modified_signal, fs=self.fs)
        psd_normalized = psd / np.sum(psd)

        # Calculating the spectral features.
        # Spectral centroid (order 1-4)
        for order in range(1, 5):
            feats.extend(self.calculate_spectral_centroid(freqs_spectrum, spectrum_magnitudes, order=order))
            feats_names.append(f"{signal_name}_spectral_centroid_order_{order}")
        # Spectral variance / spectral spread
        feats.extend(self.calculate_spectral_variance(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_var")
        # Spectral skewness
        feats.extend(self.calculate_spectral_skewness(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_skewness")
        # Spectral kurtosis
        feats.extend(self.calculate_spectral_kurtosis(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_kurtosis")
        # Median frequency of the power spectrum of a signal
        feats.extend(self.calculate_median_frequency(freqs_psd, psd))
        feats_names.append(f"{signal_name}_median_frequency")
        # Spectral bandwidth order 1 / Spectral mean deviation
        feats.extend(self.calculate_spectral_bandwidth(freqs_spectrum, spectrum_magnitudes, 1))
        feats_names.append(f"{signal_name}_spectral_mean_deviation")
        # Spectral bandwidth order 2 / Spectral standard deviation
        feats.extend(self.calculate_spectral_bandwidth(freqs_spectrum, spectrum_magnitudes, 2))
        feats_names.append(f"{signal_name}_spectral_std")
        # Spectral bandwidth order 3
        feats.extend(self.calculate_spectral_bandwidth(freqs_spectrum, spectrum_magnitudes, 3))
        feats_names.append(f"{signal_name}_spectral_bandwidth_order_3")
        # Spectral bandwidth order 4
        feats.extend(self.calculate_spectral_bandwidth(freqs_spectrum, spectrum_magnitudes, 4))
        feats_names.append(f"{signal_name}_spectral_bandwidth_order_4")
        # Spectral mean absolute deviation
        feats.extend(self.calculate_spectral_absolute_deviation(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_abs_deviation_order_1")
        # Spectral mean absolute deviation order 3
        feats.extend(self.calculate_spectral_absolute_deviation(freqs_spectrum, spectrum_magnitudes, order=3))
        feats_names.append(f"{signal_name}_spectral_abs_deviation_order_3")
        # Spectral linear slope for spectrum
        feats.extend(self.calculate_spectral_slope_linear(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectrum_linear_slope")
        # Spectral logarithmic slope for spectrum
        feats.extend(self.calculate_spectral_slope_logarithmic(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectrum_logarithmic_slope")
        # Spectral linear slope for psd
        feats.extend(self.calculate_spectral_slope_linear(freqs_psd, psd))
        feats_names.append(f"{signal_name}_power_spectrum_linear_slope")
        # Spectral logarithmic slope for psd
        feats.extend(self.calculate_spectral_slope_logarithmic(freqs_psd, psd))
        feats_names.append(f"{signal_name}_power_spectrum_logarithmic_slope")
        # Spectral flatness
        feats.extend(self.calculate_spectral_flatness(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_flatness")
        # Spectral peaks of power spectral density
        feats.extend(self.calculate_peak_frequencies(freqs_psd, psd))
        for rank in range(1, self.n_dom_freqs+1):
            feats_names.append(f"{signal_name}_peak_freq_{rank}")
        # Spectral edge frequency for different thresholds
        feats.extend(self.calculate_spectral_edge_frequency(freqs_psd, psd))
        for threshold in self.cumulative_power_thresholds:
            feats_names.append(f"{signal_name}_edge_freq_thresh_{threshold}")
        # Spectral band power for different bands
        feats.extend(self.calculate_band_power(freqs_psd, psd))
        feats_names.append(f"{signal_name}_spectral_total_power")
        for band in self.f_bands:
            feats_names.append(f"{signal_name}_spectral_abs_power_band_{str(band)}")
            feats_names.append(f"{signal_name}_spectral_rel_power_band_{str(band)}")
        # Spectral entropy
        feats.extend(self.calculate_spectral_entropy(psd))
        feats_names.append(f"{signal_name}_spectral_entropy")
        # Spectral contrast for different bands
        feats.extend(self.calculate_spectral_contrast(freqs_psd, psd))
        for band in self.f_bands:
            feats_names.append(f"{signal_name}_spectral_contrast_band_{str(band)}")
        # Spectral coefficient of variation
        feats.extend(self.calculate_spectral_cov(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_coefficient_of_variation")
        # Spectral flux
        feats.extend(self.calculate_spectral_flux(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_flux")
        # Returning the spectral features and the feature names
        return np.array(feats), feats_names

    def calculate_spectral_centroid(self, freqs, magnitudes, order=1):
        spectral_centroid = np.sum(magnitudes * (freqs ** order)) / np.sum(magnitudes)
        return np.array([spectral_centroid])

    def calculate_spectral_variance(self, freqs, magnitudes):
        # AKA spectral spread
        mean_frequency = self.calculate_spectral_centroid(freqs, magnitudes)
        spectral_variance = np.sum(((freqs - mean_frequency) ** 2) * magnitudes) / np.sum(magnitudes)
        return np.array([spectral_variance])

    def calculate_spectral_skewness(self, freqs, magnitudes):
        mu1 = self.calculate_spectral_centroid(freqs, magnitudes, order=1)
        mu2 = self.calculate_spectral_centroid(freqs, magnitudes, order=2)
        spectral_skewness = np.sum(magnitudes * (freqs - mu1) ** 3) / (np.sum(magnitudes) * mu2 ** 3)
        return spectral_skewness

    def calculate_spectral_kurtosis(self, freqs, magnitudes):
        mu1 = self.calculate_spectral_centroid(freqs, magnitudes, order=1)
        mu2 = self.calculate_spectral_centroid(freqs, magnitudes, order=2)
        spectral_kurtosis = np.sum(magnitudes * (freqs - mu1) ** 4) / (np.sum(magnitudes) * mu2 ** 4)
        return spectral_kurtosis

    def calculate_median_frequency(self, freqs, psd):
        # Calculate the cumulative distribution function (CDF) of the PSD
        cdf = np.cumsum(psd)
        median_freq = freqs[np.searchsorted(cdf, cdf[-1] / 2)]
        return np.array([median_freq])

    def calculate_spectral_flatness(self, magnitudes):
        # AKA Wiener's entropy.
        spectral_flatness = np.exp(np.mean(np.log(magnitudes))) / np.mean(magnitudes)
        return np.array([spectral_flatness])

    def calculate_spectral_slope_logarithmic(self, freqs, magnitudes):
        slope = np.polyfit(freqs, np.log(magnitudes), 1)[0]
        return np.array([slope])

    def calculate_spectral_slope_linear(self, freqs, magnitudes):
        slope = np.polyfit(freqs, magnitudes, 1)[0]
        return np.array([slope])

    def calculate_peak_frequencies(self, freqs, psd):
        peak_frequencies = freqs[np.argsort(psd)[-self.n_dom_freqs:][::-1]]
        return np.array(peak_frequencies)

    def calculate_spectral_edge_frequency(self, freqs, psd):
        # A special case would be roll-off frequency (threshold = .85)
        feats = []
        cumulative_power = np.cumsum(psd) / np.sum(psd)
        for threshold in self.cumulative_power_thresholds:
            feats.append(freqs[np.argmax(cumulative_power >= threshold)])
        return np.array(feats)

    def calculate_band_power(self, freqs, psd):
        # The features array for storing the total power, band absolute powers, and band relative powers
        feats = []
        freq_res = freqs[1] - freqs[0]  # Frequency resolution
        # Calculate the total power of the signal
        try:
            feats.append(simps(psd, dx=freq_res))
        except:
            feats.append(np.nan)
        # Calculate band absolute and relative power
        for f_band in self.f_bands:
            try:
                # Keeping the frequencies within the band
                idx_band = np.logical_and(freqs >= f_band[0], freqs < f_band[1])
                # Absolute band power by integrating PSD over frequency range of interest
                feats.append(simps(psd[idx_band], dx=freq_res))
                # Relative band power
                feats.append(feats[-1] / feats[0])
            except:
                feats.extend([np.nan, np.nan])
        return np.array(feats)

    def calculate_spectral_entropy(self, psd):
        try:
            # Formula from Matlab doc
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
        except:
            spectral_entropy = np.nan
        return np.array([spectral_entropy])

    def calculate_spectral_contrast(self, freqs, psd):
        feats = []
        for f_band in self.f_bands:
            try:
                idx_band = np.logical_and(freqs >= f_band[0], freqs < f_band[1])
                peak = np.max(psd[idx_band])
                valley = np.min(psd[idx_band])
                contrast = peak - valley
                feats.append(contrast)
            except:
                feats.append(np.nan)
        return np.array(feats)

    def calculate_spectral_bandwidth(self, freqs, magnitudes, order):
        # Definition from Librosa library (with normalized magnitudes)
        # The 1st order spectral bandwidth is the same as spectral mean deviation
        # The 2nd order spectral bandwidth is the same as spectral standard deviation
        normalized_magnitudes = magnitudes / np.sum(magnitudes)
        mean_frequency = self.calculate_spectral_centroid(freqs, magnitudes)
        spectral_bandwidth = ((np.sum(((freqs - mean_frequency) ** order) * normalized_magnitudes)) ** (1 / order))
        return np.array([spectral_bandwidth])

    def calculate_spectral_absolute_deviation(self, freqs, magnitudes, order=1):
        # Definition from Librosa library
        # The even order spectral absolute deviation is the same as spectral bandwidth of the same order
        normalized_magnitudes = magnitudes / np.sum(magnitudes)
        mean_frequency = self.calculate_spectral_centroid(freqs, magnitudes)
        spectral_absolute_deviation = ((np.sum((np.abs(freqs - mean_frequency) ** order) * normalized_magnitudes)) ** (1 / order))
        return np.array([spectral_absolute_deviation])

    def calculate_spectral_cov(self, freqs, magnitudes):
        mean_frequency = self.calculate_spectral_centroid(freqs, magnitudes)
        frequency_std = self.calculate_spectral_bandwidth(freqs, magnitudes, 2)
        coefficient_of_variation = (frequency_std / mean_frequency) * 100
        return coefficient_of_variation

    def calculate_spectral_flux(self, magnitudes, order=2):
        spectral_flux = (np.sum(np.abs(np.diff(magnitudes)) ** order)) ** (1 / order)
        return np.array([spectral_flux])
