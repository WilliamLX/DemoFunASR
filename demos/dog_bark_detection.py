import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

class DogBarkDetector:
    def __init__(self):
        self.bark_frequency_range = (200, 2000)  # Hz - typical dog bark frequencies
        self.bark_duration_range = (0.1, 2.0)    # seconds
        
    def load_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            return y, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None
    
    def extract_features(self, y, sr):
        """Extract audio features that might indicate dog barks"""
        features = {}
        
        # 1. Spectral features
        # Mel-frequency cepstral coefficients (MFCC)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # 2. Spectral centroid (brightness of sound)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # 3. Spectral rolloff (frequency below which 85% of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # 4. Zero crossing rate (measure of noisiness)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
        
        # 5. Root mean square energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 6. Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        return features
    
    def detect_barks_by_frequency(self, y, sr, threshold=0.1):
        """Detect potential barks based on frequency analysis"""
        # Compute spectrogram
        D = librosa.stft(y)
        frequencies = librosa.fft_frequencies(sr=sr)
        
        # Find frequency bins in bark range
        bark_mask = (frequencies >= self.bark_frequency_range[0]) & (frequencies <= self.bark_frequency_range[1])
        
        # Calculate energy in bark frequency range
        bark_energy = np.sum(np.abs(D[bark_mask, :])**2, axis=0)
        total_energy = np.sum(np.abs(D)**2, axis=0)
        
        # Normalize
        bark_ratio = bark_energy / (total_energy + 1e-10)
        
        # Find segments with high bark energy
        bark_segments = bark_ratio > threshold
        
        return bark_segments, bark_ratio
    
    def detect_barks_by_amplitude(self, y, sr, window_size=0.1, threshold=0.1):
        """Detect potential barks based on amplitude analysis"""
        # Convert window size to samples
        window_samples = int(window_size * sr)
        
        # Calculate RMS energy in windows
        rms_energy = []
        for i in range(0, len(y), window_samples):
            window = y[i:i+window_samples]
            if len(window) > 0:
                rms = np.sqrt(np.mean(window**2))
                rms_energy.append(rms)
        
        rms_energy = np.array(rms_energy)
        
        # Normalize
        rms_normalized = rms_energy / (np.max(rms_energy) + 1e-10)
        
        # Find segments above threshold
        bark_segments = rms_normalized > threshold
        
        return bark_segments, rms_normalized
    
    def detect_barks_combined(self, y, sr, freq_threshold=0.1, amp_threshold=0.1):
        """Combine frequency and amplitude analysis for better detection"""
        # Frequency-based detection
        freq_segments, freq_ratio = self.detect_barks_by_frequency(y, sr, freq_threshold)
        
        # Amplitude-based detection
        amp_segments, amp_ratio = self.detect_barks_by_amplitude(y, sr, threshold=amp_threshold)
        
        # Combine both methods (logical AND)
        combined_segments = freq_segments & amp_segments[:len(freq_segments)]
        
        return combined_segments, freq_ratio, amp_ratio
    
    def analyze_audio(self, file_path, plot=True):
        """Complete analysis of audio file for dog barks"""
        print(f"Analyzing: {file_path}")
        
        # Load audio
        y, sr = self.load_audio(file_path)
        if y is None:
            return None
        
        print(f"Audio loaded: {len(y)/sr:.2f} seconds, {sr} Hz")
        
        # Extract features
        features = self.extract_features(y, sr)
        print("\nAudio Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        # Detect barks
        bark_segments, freq_ratio, amp_ratio = self.detect_barks_combined(y, sr)
        
        # Count potential barks
        bark_count = np.sum(bark_segments)
        total_segments = len(bark_segments)
        bark_percentage = (bark_count / total_segments) * 100
        
        print(f"\nBark Detection Results:")
        print(f"  Potential bark segments: {bark_count}/{total_segments}")
        print(f"  Bark percentage: {bark_percentage:.2f}%")
        
        if plot:
            self.plot_analysis(y, sr, bark_segments, freq_ratio, amp_ratio)
        
        return {
            'features': features,
            'bark_segments': bark_segments,
            'bark_count': bark_count,
            'bark_percentage': bark_percentage,
            'freq_ratio': freq_ratio,
            'amp_ratio': amp_ratio
        }
    
    def plot_analysis(self, y, sr, bark_segments, freq_ratio, amp_ratio):
        """Plot the analysis results"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Plot 1: Waveform
        time = np.linspace(0, len(y)/sr, len(y))
        axes[0].plot(time, y)
        axes[0].set_title('Audio Waveform')
        axes[0].set_ylabel('Amplitude')
        
        # Plot 2: Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
        axes[1].set_title('Spectrogram')
        
        # Plot 3: Frequency ratio
        time_freq = np.linspace(0, len(y)/sr, len(freq_ratio))
        axes[2].plot(time_freq, freq_ratio)
        axes[2].set_title('Bark Frequency Ratio')
        axes[2].set_ylabel('Ratio')
        
        # Plot 4: Amplitude ratio
        time_amp = np.linspace(0, len(y)/sr, len(amp_ratio))
        axes[3].plot(time_amp, amp_ratio)
        axes[3].set_title('Bark Amplitude Ratio')
        axes[3].set_ylabel('Ratio')
        axes[3].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.show()

def main():
    detector = DogBarkDetector()
    
    # Example usage
    # Replace with your MP3 file path
    audio_file = "/Users/william/Work/VoiceData/data/20250616120811045.aac"
    
    try:
        results = detector.analyze_audio(audio_file)
        if results:
            print(f"\nAnalysis complete!")
            print(f"Bark detection confidence: {results['bark_percentage']:.2f}%")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 