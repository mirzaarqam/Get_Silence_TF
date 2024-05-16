import base64
import io
from multiprocessing import Process
from flask import Flask, render_template, request, redirect, url_for
from matplotlib import pyplot as plt
from pydub import AudioSegment
import numpy as np
import os
import librosa
import pandas as pd
import soundfile as sf
from extract_audio import get_audio_silence

app = Flask(__name__)

def get_audio_db_with_librosa(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Calculate dB
    db = librosa.amplitude_to_db(np.abs(y), ref=np.max)

    # Save dB results to CSV
    #df = pd.DataFrame({'dB': db})
    #df.to_csv("audio_db.csv", index=False)

    # Return the maximum dB value
    #print(db)
    #print(np.max(db))
    #return np.max(db)
    return np.mean(db)


def get_db_from_pydub(audio_file_path):
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

#======== Librosa Audio Chunks ======================================
def plot_audio_wave_decibels(audio_path):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    num_chunks = int(duration)
    print(num_chunks)
    print(y, sr)
    images = []
    avg_db_list = []

    output_folder = "audiochunk"

    for i in range(num_chunks):
        start = i * sr
        end = (i + 1) * sr
        chunk = y[start:end]
        db = librosa.amplitude_to_db(chunk, ref=np.max)
        save_chunk = 0

        #db_normalized = db
        #sorting
        db_normalized = np.sort(db)
        print("db normalized", db_normalized)
        # Reshape the array to have 441 rows and 50 columns
        db_reshaped = db_normalized.reshape((441, -1))
        print("db reshaped", db_reshaped)
        # Take the mean along the columns (axis=1)
        _50_norm_result = np.mean(db_reshaped, axis=1)

        print("db normalized file 50", _50_norm_result)

        for j in _50_norm_result:
            if j >= -5:
                save_chunk = save_chunk + 1
        df = pd.DataFrame(db)

        csv_filename = os.path.join(output_folder, f"db_data_{i+1}.csv")
        df.to_csv(csv_filename, index=False)

        print(save_chunk)
        if save_chunk > 0:
            chunk_filename = os.path.join(output_folder, f"chunk_{i + 1}.wav")
            # librosa.output.write_wav(chunk_filename, chunk, sr)
            sf.write(chunk_filename, chunk, sr)

        #dbpy = is_silent(chunk)
        print("chunk db", db)
        #print(dbpy)


        avg_db = np.mean(db)
        avg_db_list.append(avg_db)

        plt.figure(figsize=(12, 4))
        #plt.ylim(-0.8, 0.8)
        librosa.display.waveshow(db, sr=sr, x_axis='time')

        plt.title(f"Audio Wave in Decibels - Second {i+1} (Avg: {avg_db:.2f} dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (dB)")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        img_data = base64.b64encode(buf.read()).decode('utf-8')
        images.append(img_data)

        plt.close()


    return images, avg_db_list

#====================================================================

#============== Pydub Audio Chunks ==================================

def calculate_avg_db(audio_path):
    chunk_size = 1000
    # Load audio file
    audio = AudioSegment.from_file(audio_path)

    # Convert to numpy array
    audio_array = np.array(audio.get_array_of_samples())

    # Calculate number of chunks
    num_chunks = len(audio_array) // chunk_size

    # Initialize list to store average dB values
    avg_db_list = []

    # Iterate over chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = audio_array[start:end]
        db = 20 * np.log10(np.max(np.abs(chunk)) / (2 ** 15))  # Calculate dB
        avg_db_list.append(db)

    return avg_db_list
#====================================================================

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
        #get_audio_DB = is_silent(file_path) //pydub function call

        _silent = get_audio_silence(file_path, threshold)
        original_audio_db = get_db_from_pydub(file_path)
        chunk_audio_db = get_db_from_pydub("uploads/output_audio.wav")
        if _silent == "silent":
            silent_con += 1

        result_data.append({
            'audio_name': file.filename,
            'audio_db': original_audio_db,
            'chunks_audio_db': chunk_audio_db,
            'threshold': threshold,
            'is_silent': _silent,
        })


    not_silent = total_audio - silent_con
    return render_template('result.html', result=result_data, total=total_audio, silent=silent_con, nosilent=not_silent)

    """
        get_audio_DB = get_audio_db_with_librosa(file_path) #librosa function call
        print(get_audio_DB)
        if get_audio_DB <= threshold:
            _silent = "Silent"
            silent_con += 1
        else:
            _silent = "Not Silent"

        audio_name = file.filename
        audio_path = file_path
        waveform_images, chunks_avg_db = plot_audio_wave_decibels(file_path)

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
        #==================================================

        chunks_audios = [np.mean(chunks_avg_db), chunks_avg_db]

        result_data.append({
            'audio_name': audio_name,
            'audio_db': get_audio_DB,
            'chunks_audio_db': chunks_audios,
            'threshold': threshold,
            'is_silent': _silent,
            'graph_data': graph_data,
            'audio_graph_data': waveform_images
        })
    not_silent = total_audio - silent_con
    return render_template('result.html', result=result_data, total=total_audio, silent=silent_con, nosilent=not_silent)
    """
#================ Audio Maps With Decible ==============================

#=======================================================================





def run_flask_app():
    # app.run(debug=True)
    app.run(host='10.22.202.81', port=5002)

if __name__ == "__main__":
    p = Process(target=run_flask_app())
    p.start()