import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def simple_bark_detection(audio_file, plot=True):
    """
    Simple dog bark detection using basic audio analysis
    """
    print(f"Analyzing: {audio_file}")
    
    # Load audio
    try:
        y, sr = librosa.load(audio_file, sr=None)
        print(f"Audio loaded: {len(y)/sr:.2f} seconds, {sr} Hz")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None
    
    # 1. Basic amplitude analysis
    print("\n=== Amplitude Analysis ===")
    rms = librosa.feature.rms(y=y)[0]
    max_amplitude = np.max(np.abs(y))
    avg_amplitude = np.mean(np.abs(y))
    
    print(f"Max amplitude: {max_amplitude:.4f}")
    print(f"Average amplitude: {avg_amplitude:.4f}")
    print(f"RMS energy - Mean: {np.mean(rms):.4f}, Std: {np.std(rms):.4f}")
    
    # 2. Frequency analysis
    print("\n=== Frequency Analysis ===")
    # Dog barks typically have frequencies between 200-2000 Hz
    bark_freq_min, bark_freq_max = 200, 2000
    
    # Compute spectrogram
    D = librosa.stft(y)
    frequencies = librosa.fft_frequencies(sr=sr)
    
    # Find frequency bins in bark range
    bark_mask = (frequencies >= bark_freq_min) & (frequencies <= bark_freq_max)
    
    # Calculate energy in bark frequency range
    bark_energy = np.sum(np.abs(D[bark_mask, :])**2, axis=0)
    total_energy = np.sum(np.abs(D)**2, axis=0)
    
    # Normalize
    bark_ratio = bark_energy / (total_energy + 1e-10)
    
    print(f"Bark frequency range: {bark_freq_min}-{bark_freq_max} Hz")
    print(f"Average bark energy ratio: {np.mean(bark_ratio):.4f}")
    print(f"Max bark energy ratio: {np.max(bark_ratio):.4f}")
    
    # 3. Detect potential bark segments
    print("\n=== Bark Detection ===")
    
    # Threshold-based detection
    amplitude_threshold = 0.1 * max_amplitude
    frequency_threshold = 0.1
    
    # Find segments with high amplitude
    high_amp_segments = np.abs(y) > amplitude_threshold
    
    # Find segments with high bark frequency content
    # We need to interpolate bark_ratio to match the audio length
    bark_ratio_interp = np.interp(
        np.linspace(0, 1, len(y)), 
        np.linspace(0, 1, len(bark_ratio)), 
        bark_ratio
    )
    high_freq_segments = bark_ratio_interp > frequency_threshold
    
    # Combine both conditions
    bark_segments = high_amp_segments & high_freq_segments
    
    # Count bark events
    bark_count = np.sum(bark_segments)
    total_samples = len(bark_segments)
    bark_percentage = (bark_count / total_samples) * 100
    
    print(f"Potential bark segments: {bark_count}/{total_samples}")
    print(f"Bark percentage: {bark_percentage:.2f}%")
    
    # 4. Spectral features
    print("\n=== Spectral Features ===")
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    print(f"Spectral centroid - Mean: {np.mean(spectral_centroid):.1f} Hz")
    print(f"Spectral rolloff - Mean: {np.mean(spectral_rolloff):.1f} Hz")
    print(f"Zero crossing rate - Mean: {np.mean(zero_crossing_rate):.4f}")
    
    # 5. Bark classification
    print("\n=== Bark Classification ===")
    
    # Simple rule-based classification
    bark_score = 0
    
    # Amplitude score
    if max_amplitude > 0.5:
        bark_score += 2
    elif max_amplitude > 0.3:
        bark_score += 1
    
    # Frequency score
    if np.mean(bark_ratio) > 0.3:
        bark_score += 2
    elif np.mean(bark_ratio) > 0.1:
        bark_score += 1
    
    # Spectral centroid score (dog barks are typically bright sounds)
    if np.mean(spectral_centroid) > 2000:
        bark_score += 1
    
    # Final classification
    if bark_score >= 4:
        classification = "HIGH - Likely contains dog barks"
    elif bark_score >= 2:
        classification = "MEDIUM - May contain dog barks"
    else:
        classification = "LOW - Unlikely to contain dog barks"
    
    print(f"Bark score: {bark_score}/5")
    print(f"Classification: {classification}")
    
    if plot:
        plot_analysis(y, sr, bark_segments, bark_ratio_interp)
    
    return {
        'bark_score': bark_score,
        'classification': classification,
        'bark_percentage': bark_percentage,
        'max_amplitude': max_amplitude,
        'bark_ratio_mean': np.mean(bark_ratio)
    }

def plot_analysis(y, sr, bark_segments, bark_ratio):
    """Plot the analysis results"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot 1: Waveform with bark segments highlighted
    time = np.linspace(0, len(y)/sr, len(y))
    axes[0].plot(time, y, 'b-', alpha=0.7, label='Audio')
    
    # Highlight bark segments
    bark_times = time[bark_segments]
    bark_amplitudes = y[bark_segments]
    axes[0].scatter(bark_times, bark_amplitudes, c='red', s=10, alpha=0.8, label='Potential barks')
    
    axes[0].set_title('Audio Waveform with Bark Detection')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    
    # Plot 2: Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title('Spectrogram')
    
    # Plot 3: Bark frequency ratio over time
    time_freq = np.linspace(0, len(y)/sr, len(bark_ratio))
    axes[2].plot(time_freq, bark_ratio, 'g-', label='Bark frequency ratio')
    axes[2].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Threshold')
    axes[2].set_title('Bark Frequency Ratio Over Time')
    axes[2].set_ylabel('Ratio')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Example usage
    audio_file = "/Users/william/Work/VoiceData/data/20250616124712234.aac"
    
    try:
        results = simple_bark_detection(audio_file)
        if results:
            print(f"\n=== Summary ===")
            print(f"File: {audio_file}")
            print(f"Bark Score: {results['bark_score']}/5")
            print(f"Classification: {results['classification']}")
            print(f"Bark Percentage: {results['bark_percentage']:.2f}%")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 