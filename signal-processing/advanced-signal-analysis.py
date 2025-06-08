#!/usr/bin/env python3
"""
Advanced Signal Processing and Analysis

Comprehensive signal processing toolkit for data science applications,
covering time series analysis, frequency domain analysis, filter design,
spectral analysis, and advanced signal processing techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, fft
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.optimize import curve_fit
import scipy.stats as stats
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from scipy.signal import stft, istft, spectrogram
    ADVANCED_SCIPY_AVAILABLE = True
except ImportError:
    ADVANCED_SCIPY_AVAILABLE = False

class SignalAnalyzer:
    """Comprehensive signal analysis toolkit"""
    
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def load_signal(self, data, time=None):
        """Load signal data"""
        self.signal = np.array(data)
        if time is None:
            self.time = np.arange(len(self.signal)) / self.fs
        else:
            self.time = np.array(time)
        
        return self.signal, self.time
    
    def basic_statistics(self):
        """Calculate basic signal statistics"""
        stats = {
            'length': len(self.signal),
            'duration': len(self.signal) / self.fs,
            'mean': np.mean(self.signal),
            'std': np.std(self.signal),
            'var': np.var(self.signal),
            'min': np.min(self.signal),
            'max': np.max(self.signal),
            'rms': np.sqrt(np.mean(self.signal**2)),
            'energy': np.sum(self.signal**2),
            'power': np.mean(self.signal**2)
        }
        
        # Signal-to-noise ratio estimate
        noise_floor = np.std(self.signal - signal.medfilt(self.signal, 5))
        stats['snr_estimate'] = 20 * np.log10(stats['rms'] / noise_floor)
        
        return stats
    
    def frequency_analysis(self):
        """Comprehensive frequency domain analysis"""
        # FFT analysis
        N = len(self.signal)
        frequencies = fft.fftfreq(N, 1/self.fs)[:N//2]
        fft_magnitude = np.abs(fft.fft(self.signal))[:N//2]
        fft_phase = np.angle(fft.fft(self.signal))[:N//2]
        
        # Power spectral density
        f_psd, psd = signal.periodogram(self.signal, self.fs)
        
        # Spectral centroid (center of mass of spectrum)
        spectral_centroid = np.sum(frequencies * fft_magnitude) / np.sum(fft_magnitude)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * fft_magnitude) / np.sum(fft_magnitude))
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitude)
        dominant_frequency = frequencies[dominant_freq_idx]
        
        return {
            'frequencies': frequencies,
            'fft_magnitude': fft_magnitude,
            'fft_phase': fft_phase,
            'psd_frequencies': f_psd,
            'psd': psd,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'dominant_frequency': dominant_frequency,
            'total_power': np.sum(psd)
        }
    
    def find_signal_peaks(self, height=None, prominence=None, distance=None):
        """Find and analyze signal peaks"""
        # Find peaks
        peaks, properties = find_peaks(
            self.signal, 
            height=height, 
            prominence=prominence, 
            distance=distance
        )
        
        # Calculate peak properties
        prominences = peak_prominences(self.signal, peaks)[0]
        widths = peak_widths(self.signal, peaks, rel_height=0.5)[0]
        
        peak_info = {
            'peak_indices': peaks,
            'peak_values': self.signal[peaks],
            'peak_times': self.time[peaks],
            'prominences': prominences,
            'widths': widths,
            'count': len(peaks)
        }
        
        if len(peaks) > 0:
            peak_info['mean_prominence'] = np.mean(prominences)
            peak_info['mean_width'] = np.mean(widths)
            peak_info['peak_frequency'] = len(peaks) / (self.time[-1] - self.time[0])
        
        return peak_info
    
    def time_frequency_analysis(self, window='hann', nperseg=256):
        """Time-frequency analysis using STFT"""
        if not ADVANCED_SCIPY_AVAILABLE:
            raise ImportError("Advanced scipy functions not available")
        
        # Short-time Fourier transform
        f, t, Zxx = stft(self.signal, self.fs, window=window, nperseg=nperseg)
        
        # Spectrogram
        f_spec, t_spec, Sxx = spectrogram(self.signal, self.fs, window=window, nperseg=nperseg)
        
        return {
            'stft_frequencies': f,
            'stft_times': t,
            'stft_complex': Zxx,
            'stft_magnitude': np.abs(Zxx),
            'stft_phase': np.angle(Zxx),
            'spectrogram_frequencies': f_spec,
            'spectrogram_times': t_spec,
            'spectrogram': Sxx
        }

class FilterDesigner:
    """Advanced digital filter design and application"""
    
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
    
    def design_lowpass(self, cutoff, order=5, filter_type='butter'):
        """Design lowpass filter"""
        nyquist_cutoff = cutoff / self.nyquist
        
        if filter_type == 'butter':
            b, a = signal.butter(order, nyquist_cutoff, btype='low')
        elif filter_type == 'cheby1':
            b, a = signal.cheby1(order, 1, nyquist_cutoff, btype='low')
        elif filter_type == 'ellip':
            b, a = signal.ellip(order, 1, 40, nyquist_cutoff, btype='low')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        return b, a
    
    def design_highpass(self, cutoff, order=5, filter_type='butter'):
        """Design highpass filter"""
        nyquist_cutoff = cutoff / self.nyquist
        
        if filter_type == 'butter':
            b, a = signal.butter(order, nyquist_cutoff, btype='high')
        elif filter_type == 'cheby1':
            b, a = signal.cheby1(order, 1, nyquist_cutoff, btype='high')
        elif filter_type == 'ellip':
            b, a = signal.ellip(order, 1, 40, nyquist_cutoff, btype='high')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        return b, a
    
    def design_bandpass(self, low_cutoff, high_cutoff, order=5, filter_type='butter'):
        """Design bandpass filter"""
        low_nyquist = low_cutoff / self.nyquist
        high_nyquist = high_cutoff / self.nyquist
        
        if filter_type == 'butter':
            b, a = signal.butter(order, [low_nyquist, high_nyquist], btype='band')
        elif filter_type == 'cheby1':
            b, a = signal.cheby1(order, 1, [low_nyquist, high_nyquist], btype='band')
        elif filter_type == 'ellip':
            b, a = signal.ellip(order, 1, 40, [low_nyquist, high_nyquist], btype='band')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        return b, a
    
    def design_notch(self, frequency, quality_factor=30):
        """Design notch filter for specific frequency"""
        w0 = frequency / self.nyquist
        b, a = signal.iirnotch(w0, quality_factor)
        return b, a
    
    def apply_filter(self, signal_data, b, a, method='filtfilt'):
        """Apply designed filter to signal"""
        if method == 'filtfilt':
            # Zero-phase filtering
            filtered_signal = signal.filtfilt(b, a, signal_data)
        elif method == 'lfilter':
            # Standard filtering
            filtered_signal = signal.lfilter(b, a, signal_data)
        else:
            raise ValueError(f"Unsupported filtering method: {method}")
        
        return filtered_signal
    
    def analyze_filter_response(self, b, a, worN=8192):
        """Analyze filter frequency response"""
        w, h = signal.freqz(b, a, worN=worN)
        frequencies = w * self.fs / (2 * np.pi)
        
        magnitude_db = 20 * np.log10(abs(h))
        phase_deg = np.angle(h) * 180 / np.pi
        
        # Group delay
        _, group_delay = signal.group_delay((b, a), w)
        
        return {
            'frequencies': frequencies,
            'magnitude_db': magnitude_db,
            'phase_deg': phase_deg,
            'group_delay': group_delay,
            'magnitude_linear': abs(h)
        }

class WaveletAnalyzer:
    """Wavelet analysis for time-frequency decomposition"""
    
    def __init__(self):
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets not available")
    
    def continuous_wavelet_transform(self, signal_data, wavelet='cmor', scales=None):
        """Continuous Wavelet Transform"""
        if scales is None:
            scales = np.arange(1, 128)
        
        # Perform CWT
        coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet)
        
        return {
            'coefficients': coefficients,
            'scales': scales,
            'frequencies': frequencies
        }
    
    def discrete_wavelet_transform(self, signal_data, wavelet='db4', levels=5):
        """Discrete Wavelet Transform"""
        # Multi-level DWT
        coeffs = pywt.wavedec(signal_data, wavelet, level=levels)
        
        # Reconstruct signal
        reconstructed = pywt.waverec(coeffs, wavelet)
        
        return {
            'coefficients': coeffs,
            'levels': levels,
            'reconstructed': reconstructed
        }
    
    def wavelet_denoising(self, signal_data, wavelet='db4', levels=5, threshold_mode='soft'):
        """Wavelet-based denoising"""
        # Decompose signal
        coeffs = pywt.wavedec(signal_data, wavelet, level=levels)
        
        # Estimate noise level using median absolute deviation
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        
        # Apply thresholding
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [pywt.threshold(i, threshold, mode=threshold_mode) for i in coeffs_thresh[1:]]
        
        # Reconstruct denoised signal
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        
        return {
            'denoised_signal': denoised,
            'noise_estimate': sigma,
            'threshold_used': threshold
        }

class AdvancedSignalProcessor:
    """Advanced signal processing techniques"""
    
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
    
    def hilbert_transform(self, signal_data):
        """Hilbert transform for analytic signal"""
        analytic_signal = signal.hilbert(signal_data)
        
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase)) / (2.0 * np.pi) * self.fs
        
        return {
            'analytic_signal': analytic_signal,
            'amplitude_envelope': amplitude_envelope,
            'instantaneous_phase': instantaneous_phase,
            'instantaneous_frequency': instantaneous_frequency
        }
    
    def empirical_mode_decomposition(self, signal_data, max_imfs=10):
        """Simple EMD implementation (placeholder for more advanced EMD)"""
        # This is a simplified version - in practice, use PyEMD library
        imfs = []
        residue = signal_data.copy()
        
        for i in range(max_imfs):
            # Extract peaks and troughs
            peaks, _ = find_peaks(residue)
            troughs, _ = find_peaks(-residue)
            
            if len(peaks) < 3 or len(troughs) < 3:
                break
            
            # Simple envelope estimation
            upper_env = np.interp(np.arange(len(residue)), peaks, residue[peaks])
            lower_env = np.interp(np.arange(len(residue)), troughs, residue[troughs])
            
            mean_env = (upper_env + lower_env) / 2
            imf = residue - mean_env
            
            imfs.append(imf)
            residue = residue - imf
            
            # Check stopping criterion
            if np.std(imf) < 0.01:
                break
        
        imfs.append(residue)  # Final residue
        
        return {
            'imfs': imfs,
            'residue': residue,
            'num_imfs': len(imfs) - 1
        }
    
    def independent_component_analysis(self, signals, n_components=None):
        """ICA for blind source separation"""
        if signals.ndim == 1:
            raise ValueError("ICA requires multiple signals (2D array)")
        
        # Standardize signals
        scaler = StandardScaler()
        signals_scaled = scaler.fit_transform(signals.T).T
        
        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=42)
        sources = ica.fit_transform(signals_scaled.T).T
        
        return {
            'independent_components': sources,
            'mixing_matrix': ica.mixing_,
            'unmixing_matrix': ica.components_
        }
    
    def principal_component_analysis(self, signals):
        """PCA for dimensionality reduction"""
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        
        # Standardize signals
        scaler = StandardScaler()
        signals_scaled = scaler.fit_transform(signals.T).T
        
        # Apply PCA
        pca = PCA()
        components = pca.fit_transform(signals_scaled.T).T
        
        return {
            'principal_components': components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'singular_values': pca.singular_values_
        }
    
    def cross_correlation_analysis(self, signal1, signal2, mode='full'):
        """Cross-correlation analysis between two signals"""
        # Cross-correlation
        correlation = signal.correlate(signal1, signal2, mode=mode)
        
        # Find lag of maximum correlation
        max_corr_idx = np.argmax(np.abs(correlation))
        
        if mode == 'full':
            lags = np.arange(-len(signal2) + 1, len(signal1))
        elif mode == 'valid':
            lags = np.arange(len(signal1) - len(signal2) + 1)
        else:  # mode == 'same'
            lags = np.arange(-len(signal2)//2, len(signal1) - len(signal2)//2)
        
        optimal_lag = lags[max_corr_idx]
        max_correlation = correlation[max_corr_idx]
        
        # Normalized cross-correlation
        norm_correlation = correlation / np.sqrt(np.dot(signal1, signal1) * np.dot(signal2, signal2))
        
        return {
            'correlation': correlation,
            'normalized_correlation': norm_correlation,
            'lags': lags,
            'optimal_lag': optimal_lag,
            'max_correlation': max_correlation,
            'max_normalized_correlation': norm_correlation[max_corr_idx]
        }

class SignalQualityAssessment:
    """Signal quality assessment and validation"""
    
    def __init__(self):
        pass
    
    def detect_artifacts(self, signal_data, method='statistical'):
        """Detect artifacts in signal"""
        if method == 'statistical':
            # Statistical outlier detection
            z_scores = np.abs(stats.zscore(signal_data))
            threshold = 3
            artifacts = np.where(z_scores > threshold)[0]
            
        elif method == 'derivative':
            # Sudden changes detection
            diff_signal = np.diff(signal_data)
            threshold = np.std(diff_signal) * 5
            artifacts = np.where(np.abs(diff_signal) > threshold)[0]
            
        elif method == 'amplitude':
            # Amplitude-based detection
            amplitude_threshold = np.percentile(np.abs(signal_data), 99)
            artifacts = np.where(np.abs(signal_data) > amplitude_threshold)[0]
            
        else:
            raise ValueError(f"Unknown artifact detection method: {method}")
        
        return {
            'artifact_indices': artifacts,
            'artifact_count': len(artifacts),
            'artifact_percentage': len(artifacts) / len(signal_data) * 100
        }
    
    def signal_to_noise_ratio(self, signal_data, noise_data=None):
        """Calculate signal-to-noise ratio"""
        if noise_data is None:
            # Estimate noise from signal (simple approach)
            # Assume high-frequency components represent noise
            b, a = signal.butter(5, 0.1, btype='high')
            noise_estimate = signal.filtfilt(b, a, signal_data)
            signal_estimate = signal_data - noise_estimate
        else:
            signal_estimate = signal_data
            noise_estimate = noise_data
        
        signal_power = np.mean(signal_estimate ** 2)
        noise_power = np.mean(noise_estimate ** 2)
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return {
            'snr_linear': snr_linear,
            'snr_db': snr_db,
            'signal_power': signal_power,
            'noise_power': noise_power
        }
    
    def signal_stationarity_test(self, signal_data, window_size=None):
        """Test for signal stationarity"""
        if window_size is None:
            window_size = len(signal_data) // 4
        
        # Augmented Dickey-Fuller test (simplified version)
        # Split signal into windows and test for consistency
        n_windows = len(signal_data) // window_size
        window_means = []
        window_vars = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            window_data = signal_data[start_idx:end_idx]
            
            window_means.append(np.mean(window_data))
            window_vars.append(np.var(window_data))
        
        # Test for constant mean and variance
        mean_stability = np.std(window_means) / np.mean(np.abs(window_means))
        var_stability = np.std(window_vars) / np.mean(window_vars)
        
        # Simple stationarity score (lower is more stationary)
        stationarity_score = mean_stability + var_stability
        
        return {
            'window_means': window_means,
            'window_variances': window_vars,
            'mean_stability': mean_stability,
            'variance_stability': var_stability,
            'stationarity_score': stationarity_score,
            'is_stationary': stationarity_score < 0.5  # Threshold-based
        }

# Example usage and comprehensive analysis pipeline
def comprehensive_signal_analysis_example():
    """Comprehensive signal analysis example"""
    
    # Generate example signals
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 2, 2*fs)
    
    # Create composite signal
    signal1 = 2 * np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    signal2 = 1.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave
    signal3 = 0.5 * np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave
    noise = 0.3 * np.random.randn(len(t))
    
    composite_signal = signal1 + signal2 + signal3 + noise
    
    print("=== Comprehensive Signal Analysis ===")
    
    # Initialize analyzers
    analyzer = SignalAnalyzer(sampling_rate=fs)
    filter_designer = FilterDesigner(sampling_rate=fs)
    processor = AdvancedSignalProcessor(sampling_rate=fs)
    quality_assessor = SignalQualityAssessment()
    
    # Load signal
    analyzer.load_signal(composite_signal, t)
    
    # Basic statistics
    basic_stats = analyzer.basic_statistics()
    print(f"\nBasic Statistics:")
    print(f"Duration: {basic_stats['duration']:.2f} seconds")
    print(f"RMS: {basic_stats['rms']:.3f}")
    print(f"SNR estimate: {basic_stats['snr_estimate']:.2f} dB")
    
    # Frequency analysis
    freq_analysis = analyzer.frequency_analysis()
    print(f"\nFrequency Analysis:")
    print(f"Dominant frequency: {freq_analysis['dominant_frequency']:.2f} Hz")
    print(f"Spectral centroid: {freq_analysis['spectral_centroid']:.2f} Hz")
    print(f"Spectral bandwidth: {freq_analysis['spectral_bandwidth']:.2f} Hz")
    
    # Peak detection
    peaks = analyzer.find_signal_peaks(prominence=0.5)
    print(f"\nPeak Analysis:")
    print(f"Number of peaks: {peaks['count']}")
    if peaks['count'] > 0:
        print(f"Mean prominence: {peaks['mean_prominence']:.3f}")
        print(f"Peak frequency: {peaks['peak_frequency']:.2f} Hz")
    
    # Filter design and application
    print(f"\nFilter Analysis:")
    
    # Lowpass filter
    b_lp, a_lp = filter_designer.design_lowpass(cutoff=80, order=5)
    filtered_signal = filter_designer.apply_filter(composite_signal, b_lp, a_lp)
    
    # Analyze filter response
    filter_response = filter_designer.analyze_filter_response(b_lp, a_lp)
    print(f"Lowpass filter designed with cutoff at 80 Hz")
    
    # Signal quality assessment
    quality_results = quality_assessor.signal_to_noise_ratio(composite_signal)
    print(f"\nSignal Quality:")
    print(f"SNR: {quality_results['snr_db']:.2f} dB")
    
    artifacts = quality_assessor.detect_artifacts(composite_signal)
    print(f"Artifacts detected: {artifacts['artifact_percentage']:.2f}% of signal")
    
    stationarity = quality_assessor.signal_stationarity_test(composite_signal)
    print(f"Signal stationarity: {'Yes' if stationarity['is_stationary'] else 'No'}")
    print(f"Stationarity score: {stationarity['stationarity_score']:.3f}")
    
    # Advanced processing
    print(f"\nAdvanced Processing:")
    
    # Hilbert transform
    hilbert_results = processor.hilbert_transform(composite_signal)
    mean_inst_freq = np.mean(hilbert_results['instantaneous_frequency'])
    print(f"Mean instantaneous frequency: {mean_inst_freq:.2f} Hz")
    
    # Cross-correlation with original components
    corr_results = processor.cross_correlation_analysis(composite_signal, signal1)
    print(f"Cross-correlation with 10Hz component: {corr_results['max_normalized_correlation']:.3f}")
    
    # Wavelet analysis (if available)
    if PYWT_AVAILABLE:
        wavelet_analyzer = WaveletAnalyzer()
        denoised_results = wavelet_analyzer.wavelet_denoising(composite_signal)
        print(f"Wavelet denoising - noise estimate: {denoised_results['noise_estimate']:.3f}")
    
    # Time-frequency analysis
    if ADVANCED_SCIPY_AVAILABLE:
        tf_analysis = analyzer.time_frequency_analysis()
        print(f"Time-frequency analysis completed - STFT shape: {tf_analysis['stft_magnitude'].shape}")
    
    return {
        'signal': composite_signal,
        'time': t,
        'basic_stats': basic_stats,
        'frequency_analysis': freq_analysis,
        'peaks': peaks,
        'filtered_signal': filtered_signal,
        'quality_results': quality_results,
        'artifacts': artifacts,
        'stationarity': stationarity
    }

if __name__ == "__main__":
    # Run comprehensive analysis
    results = comprehensive_signal_analysis_example()
    
    print("\n" + "="*50)
    print("Signal processing analysis complete!")
    print("="*50) 