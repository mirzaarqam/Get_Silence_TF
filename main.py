import base64
import io

from flask import Flask, render_template, request
from matplotlib import pyplot as plt
from pydub import AudioSegment
import numpy as np
import os
import librosa



app = Flask(__name__)


def is_silent(audio_file_path):
    # Load audio file using appropriate library based on file extension
    extension = os.path.splitext(audio_file_path)[1].lower()

    if extension == ".wav":
        audio = AudioSegment.from_wav(audio_file_path)
    elif extension == ".ogg":
        audio = AudioSegment.from_ogg(audio_file_path)
    elif extension == ".mp3":
        audio = AudioSegment.from_mp3(audio_file_path)
    elif extension == ".m4a":
        audio = AudioSegment.from_file(audio_file_path, format="m4a")
    else:
        raise ValueError("Unsupported file format")

    my_sound = AudioSegment.from_file(audio_file_path)
    audio_sample = np.array(my_sound.get_array_of_samples())
    dBFS = 20 * np.log10(np.max(np.abs(audio_sample)) / (2 ** 15))
    return dBFS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    threshold = float(request.form['threshold'])
    audio_files = request.files.getlist('audio_files')
    result_data = []
    silent_con = 0
    total_audio = len(audio_files)


    for file in audio_files:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        get_audio_DB = is_silent(file_path)
        if get_audio_DB <= threshold:
            _silent = "Silent"
            silent_con += 1
        else:
            _silent = "Not Silent"

        audio_name = file.filename
        #=================
        # Plot the waveform
        y, sr = librosa.load(file_path)
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr, x_axis='time')
        plt.title('Waveform (Amplitude)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        graph_data = base64.b64encode(buf.read()).decode('utf-8')
        #=================

        result_data.append({
            'audio_name': audio_name,
            'audio_db': get_audio_DB,
            'threshold': threshold,
            'is_silent': _silent,
            'graph_data': graph_data
        })
    not_silent = total_audio - silent_con
    return render_template('result.html', result=result_data, total=total_audio, silent=silent_con, nosilent=not_silent)


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='10.22.202.81', port=5002)