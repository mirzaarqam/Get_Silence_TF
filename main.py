from pydub import AudioSegment
import numpy as np
def is_silent(audio_file_path):

    my_sound = AudioSegment.from_file(audio_file_path)
    audio_sample = np.array(my_sound.get_array_of_samples())
    dBFS = 20 * np.log10(np.max(np.abs(audio_sample)) / (2 ** 15))
    threshold = -40
    return dBFS <= threshold

if __name__ == "__main__":
    audio_file_path = "silent.wav"
    if is_silent(audio_file_path):
        print("The audio file is silent.")
    else:
        print("The audio file is not silent.")