import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
import matplotlib.pyplot as plt

def detect_local_maxima(Sxx_db: np.ndarray, frequencies: np.ndarray, times: np.ndarray, threshold_value=5) -> list:
    """
    Detect local maxima (peaks) in the spectrogram for each frequency band.
    
    Args:
        Sxx_db (np.ndarray): Spectrogram data in decibels.
        frequencies (np.ndarray): Array of frequency values.
        times (np.ndarray): Array of time values.
        threshold_value (int): Threshold to filter peaks with height greater than the mean plus threshold.
        
    Returns:
        list: A list of detected local maxima in the format [(frequency, time)].
    """
    maxima = []
    for freq_idx, row in enumerate(Sxx_db):  # Iterate over each frequency band (row of the spectrogram)
        # Find peaks in the current frequency band, applying a threshold for noise reduction
        peaks, _ = find_peaks(row, height=np.mean(row) + threshold_value)
        # Append each detected peak as a (frequency, time) tuple
        maxima.extend([(frequencies[freq_idx], times[peak_idx]) for peak_idx in peaks])
    return maxima

def detect_constellations(Sxx_db: np.ndarray, frequencies: np.ndarray, times: np.ndarray, time_window: float = 0.5) -> list:
    """
    Create constellations of points in the spectrogram based on time and frequency relationships.
    
    Args:
        Sxx_db (np.ndarray): Spectrogram data in decibels.
        frequencies (np.ndarray): Array of frequency values.
        times (np.ndarray): Array of time values.
        time_window (float): Maximum allowable time difference between points in a constellation.
        
    Returns:
        list: A list of constellations in the format [(f1, t1, f2, delta_t)].
    """
    maxima = detect_local_maxima(Sxx_db, frequencies, times)  # Detect peaks in the spectrogram
    constellations = []
    for i, (f1, t1) in enumerate(maxima):  # Anchor point (f1, t1)
        for f2, t2 in maxima[i + 1:]:  # Target point (f2, t2)
            if 0 < t2 - t1 <= time_window:  # Ensure time difference is within the window
                constellations.append((f1, t1, f2, t2 - t1))  # Store the constellation
    return constellations

def plot_spectrogram_with_constellations(audio_path: str, time_window: float = 0.5) -> None:
    """
    Plot the spectrogram of the audio file and visualize constellations.
    
    Args:
        audio_path (str): Path to the input audio file.
        time_window (float): Maximum allowable time difference between points in a constellation.
    """
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Read the audio file and convert to mono if stereo
    sample_rate, audio_data = wavfile.read(audio_path)
    if len(audio_data.shape) > 1:  # Check if the audio is stereo
        audio_data = audio_data.mean(axis=1)  # Convert to mono by averaging channels

    # Generate the spectrogram
    frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate, window='hann', nperseg=1024, noverlap=512)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Convert spectrogram to decibel scale

    # Detect constellations in the spectrogram
    constellations = detect_constellations(Sxx_db, frequencies, times, time_window)
    print(f"Detected {len(constellations)} constellations.")
    print(f"Sample constellations: {constellations[:5]}")  # Print the first 5 constellations for reference

    # Plot the spectrogram
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud')
    plt.colorbar(label='Intensity [dB]')

    # Visualize the constellations on the spectrogram
    for idx, (f1, t1, f2, delta_t) in enumerate(constellations):
        plt.plot(
            [t1, t1 + delta_t],  # Time points for the anchor and target
            [f1, f2],  # Frequency points for the anchor and target
            'ro-',  # Red line with circle markers
            markersize=2,
            label='Constellation' if idx == 0 else None  # Label only the first constellation
        )

    # Add titles and labels to the plot
    plt.title('Spectrogram with Constellations')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.legend()
    plt.show()

# Usage Example
plot_spectrogram_with_constellations('test.wav', time_window=0.5)
