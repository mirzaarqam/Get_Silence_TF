from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
import torchaudio
import pickle
import sys
#sys.path.append('silero-vad-master')
import utils_vad


#++++++++++++++++++++++++++++++++++++++++ LOAD MODEL
# Define a function to load the model from the local file
# Download and save the Silero VAD model locally
#model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
#torch.save(model.state_dict(), 'silero_vad_model.pth')
#with open('silero_vad_utils.pkl', 'wb') as f:
#    pickle.dump(utils, f)
# Function to load the model and utilities from local files
# Function to load the model and utilities from local files

# Function to load the model and utilities from local files
# Function to load the model and utilities from local files
# Function to load the TorchScript JIT model and utilities from local files
def load_model_and_utils(model_path, utils_path):
    # Load the TorchScript model
    model = torch.jit.load(model_path)
    model.eval()

    # Load the utilities
    with open(utils_path, 'rb') as f:
        utils = pickle.load(f)

    return model, utils

# Load the Silero VAD model and utilities from local files
model_path = 'silero_vad_model_jit.pt'
utils_path = 'silero_vad_utils.pkl'
print("model loaded")
model, utils = load_model_and_utils(model_path, utils_path)
(get_speech_ts, _, read_audio, _, _) = utils

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Load the Silero VAD model
#model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
#(get_speech_ts, _, read_audio, _, _) = utils




def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


def detect_speech_segments(waveform, sample_rate, threshold):
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


def get_audio_silence(input_file, threshold_value):
    input_audio_file = input_file
    output_audio_file = 'uploads/output_audio.wav'

    #threshold_value = 0.6

    remove_silence_from_audio(input_audio_file, output_audio_file, threshold=threshold_value)

    audio = AudioSegment.from_file("uploads/output_audio.wav")
    duration = audio.duration_seconds

    is_silent = ""
    if duration < 1:
        is_silent = "silent"
    else:
        is_silent = "not_silent"

    return is_silent
