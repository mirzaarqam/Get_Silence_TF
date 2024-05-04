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
    result_text = []

    for file in audio_files:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        get_audio_DB = is_silent(file_path)
        if get_audio_DB <= threshold:
            result_text.append(f"{file.filename} is silent. Audio DB {get_audio_DB} and Threshold {threshold}")
        else:
            result_text.append(f"{file.filename} is not silent. Audio DB {get_audio_DB} and Threshold {threshold}")

    return render_template('result.html', result=result_text)

if __name__ == "__main__":
    app.run(debug=True)
