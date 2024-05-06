import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
audio_file = "static/12210.wav"
y, sr = librosa.load(audio_file)

# Plot the waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
