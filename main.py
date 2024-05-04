from flask import Flask, render_template, request
from pydub import AudioSegment
import numpy as np

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
    audio_file = request.files['audio_file']
    threshold = float(request.form['threshold'])
    audio_file.save('static/audio.wav')
    get_audio_DB = is_silent('static/audio.wav')
    if get_audio_DB <= threshold:
        return f"The audio file is silent. Audio DB {get_audio_DB} and Threshold {threshold}"
    else:
        return f"The audio file is not silent. Audio DB {get_audio_DB} and Threshold {threshold}"

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='10.22.202.81', port=5001)
