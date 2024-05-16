from extract_audio import get_audio_silence

file_address = "psca-lahore-15-21058-03704154124-01-05-24-1714553078.4992642.mp3"
threshold = 0.6
print(get_audio_silence(file_address, threshold))



