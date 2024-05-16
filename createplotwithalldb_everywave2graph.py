import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

def get_db(signal):
    return 20 * np.log10(np.sqrt(np.mean(np.square(signal))))

def plot_audio_wave(audio_file):
    audio = AudioSegment.from_file(audio_file)

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    # Calculate dB and split signal into chunks
    chunk_size = sample_rate // 10  # Split into 10 equal chunks
    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]

    # Plot audio wave and dB
    plt.figure(figsize=(12, 6))
    plt.plot(samples, color='blue')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')

    db_values = [get_db(chunk) for chunk in chunks]
    for i, db in enumerate(db_values):
        plt.text(i * chunk_size, min(samples), f"{db:.2f} dB", fontsize=8)

    plt.grid()

    # Plot dB values in a separate line graph
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(db_values)), db_values, color='green', marker='o', linestyle='-')
    plt.xlabel('Chunk')
    plt.ylabel('dB')
    plt.title('dB Values')
    plt.grid()

    plt.show()

if __name__ == "__main__":
    audio_file = "psca-lahore-15-21011-03211848581-01-05-24-1714552258.4990343.mp3"
    plot_audio_wave(audio_file)