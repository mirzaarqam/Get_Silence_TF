from flask import Flask, render_template, request
from pydub import AudioSegment
import numpy as np
import os

app = Flask(__name__)

def is_silent(audio_file_path):
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
        result_data.append({
            'audio_name': audio_name,
            'audio_db': get_audio_DB,
            'threshold': threshold,
            'is_silent': _silent
        })
    not_silent = total_audio - silent_con
    return render_template('result.html', result=result_data, total=total_audio, silent=silent_con, nosilent=not_silent)


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='10.22.202.81', port=5001)