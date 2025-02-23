import numpy as np
from scipy.io.wavfile import write

# Generate a 1-second sine wave at 440 Hz
sample_rate = 44100  # Samples per second
duration = 1.0       # Duration in seconds
frequency = 440.0    # Frequency of the sine wave (A4 note)

t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Time array
audio_data = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)  # Sine wave

# Save to WAV file
write("test.wav", sample_rate, audio_data)
print("WAV file 'test.wav' generated.")
