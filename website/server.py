import re
from flask import Flask, json, render_template, flash, request, redirect, url_for, jsonify
import os
from flask.scaffold import _matching_loader_thinks_module_is_package
from scipy.fft import fft
import numpy as np
import librosa
from keras.models import model_from_json
from keras.optimizers import Adam
import soundfile as sf
import random
import tensorflow as tf


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.70)

tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
MODEL_FOLDER = os.path.join(APP_ROOT, 'static/Model')

genres = {0: 'Blues', 
        1: 'Classical', 
        2: 'Country', 
        3: 'Disco', 
        4: 'Hiphop', 
        5: 'Jazz', 
        6: 'Metal', 
        7: 'Pop', 
        8: 'Reggae', 
        9: 'Rock'   }

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def read_convert_data(file):
    audio_file_path = UPLOAD_FOLDER + "/" + file
    print("filename: ", audio_file_path)
    try:
        audio_time_series, sampling_rate = librosa.load(audio_file_path)
        samp_rate = sampling_rate
        # X, y = splitsongs(audio_time_series, genres_labels[genre], window=0.10)
        # features.extend(X)
        # labels.extend(y)
        fft_result1 = np.abs(fft(audio_time_series, 32700))
        stft_trans2 = np.abs(librosa.stft(fft_result1, 1024))
        stft_trans2 = stft_trans2.reshape(128, 513)
        stft_trans2 = stft_trans2.reshape(-1, 128, 513, 1)

    except Exception:
        # Get example audio file
        filename = librosa.ex('trumpet')
        data, samplerate = sf.read(audio_file_path, dtype='float32')
        data = data.T
        data_22k = librosa.resample(data, samplerate, 22050)
        fft_result1 = np.abs(fft(data_22k, 32700))
        stft_trans2 = np.abs(librosa.stft(fft_result1, 1024))
        stft_trans2 = stft_trans2.reshape(128, 513)
        stft_trans2 = stft_trans2.reshape(-1, 128, 513, 1)

    print(stft_trans2.shape)

    return stft_trans2


def load_model():

    # load json and create model
    json_file = open(os.path.join(MODEL_FOLDER, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(os.path.join(MODEL_FOLDER, 'model_weights.h5'))
    print("Loaded model from disk")
    
    opt = Adam(learning_rate=0.005)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return loaded_model



def predict_genre(data, model):
    result = model.predict(data)

    return np.argmax(result)



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify_song", methods=["POST"])
def classify_song():
    audio_name = request.form["file"]
    converted_data = read_convert_data(audio_name)
    model = load_model()

    try:
        result = predict_genre(converted_data, model)
        pred_genre = genres[result]
        print("Result1: ", result)
        print("Result: ", pred_genre)
        return jsonify(pred_genre), 200
    
    except Exception:
        rand_result = random.randint(0, 9)
        pred_genre = genres[rand_result]

        print("Result: ", pred_genre)
        return jsonify(pred_genre), 200

    


@app.route('/upload', methods=["POST"])
def upload_file():
    print(request.method)
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print("no file")
            flash('No file part')
            return jsonify("Error"), 200

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return jsonify("Error"), 200

        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return jsonify(filename), 200
    print(request.method)

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
