from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
import torchaudio

# Load the Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
(get_speech_ts, _, read_audio, _, _) = utils




def load_audio(file_path):
    # Load audio file using torchaudio
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


def detect_speech_segments(waveform, sample_rate, threshold):
    # Use the VAD model to get timestamps of speech segments
    speech_ts = get_speech_ts(waveform, model, sampling_rate=sample_rate, threshold=threshold)
    return speech_ts


def extract_speech_segments(audio, sample_rate, speech_ts):
    speech_audio = AudioSegment.empty()
    for ts in speech_ts:
        start_ms = (ts['start'] / sample_rate) * 1000
        end_ms = (ts['end'] / sample_rate) * 1000
        speech_audio += audio[start_ms:end_ms]
    return speech_audio


def remove_silence_from_audio(file_path, output_path, threshold):
    waveform, sample_rate = load_audio(file_path)
    speech_ts = detect_speech_segments(waveform, sample_rate, threshold=threshold)
    audio = AudioSegment.from_file(file_path)
    speech_audio = extract_speech_segments(audio, sample_rate, speech_ts)
    speech_audio.export(output_path, format="wav")

# Path to the input audio file and the output file

def get_audio_silence(input_file, threshold_value):
    input_audio_file = input_file
    output_audio_file = 'uploads/output_audio.wav'

    #threshold_value = 0.6

    # Remove silence and save the speech segments
    remove_silence_from_audio(input_audio_file, output_audio_file, threshold=threshold_value)

    #Check for silence
    audio = AudioSegment.from_file("uploads/output_audio.wav")
    duration = audio.duration_seconds

    is_silent = ""
    if duration < 1:
        is_silent = "silent"
    else:
        is_silent = "not_silent"

    return is_silent
