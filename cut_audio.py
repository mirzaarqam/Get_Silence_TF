import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_audio_wave_decibels(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Calculate number of chunks
    duration = librosa.get_duration(y=y, sr=sr)
    num_chunks = int(duration)

    plt.figure(figsize=(12, 6))

    for i in range(num_chunks):
        # Extract one-second chunk
        start = i * sr
        end = (i + 1) * sr
        chunk = y[start:end]

        # Calculate decibels
        db = librosa.amplitude_to_db(chunk, ref=np.max)

        # Calculate average decibel level
        avg_db = np.mean(db)

        # Plot audio wave in decibels
        plt.subplot(num_chunks, 1, i+1)
        librosa.display.waveshow(db, sr=sr)
        plt.title(f"Audio Wave in Decibels - Second {i+1} (Avg: {avg_db:.2f} dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (dB)")

    plt.tight_layout()
    plt.show()

# Provide the path to your audio file
audio_path = "myaudio.wav"
plot_audio_wave_decibels(audio_path)
