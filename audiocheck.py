from pydub import AudioSegment
cut_audio_duration = AudioSegment.from_file("uploads/output_audio.wav")
cut_duration = cut_audio_duration.duration_seconds
print(cut_duration)