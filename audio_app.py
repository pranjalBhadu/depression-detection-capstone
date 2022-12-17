import os
import tensorflow as tf
import numpy as np
from numpy.lib import stride_tricks
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, redirect, flash, url_for

audio_model = tf.keras.models.load_model('audio_model.h5')

app = Flask(__name__)

@app.route("/audio")
def audio_page():
    return render_template('audio.html')

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    frame = np.floor((frameSize)/2.0)
    frame = frame.astype(np.int64)
    size = np.zeros(frame)
    samples = np.append((size), sig)
    # cols for windowing
    cols = int(np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                      samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

def logscale_spec(spec, sr=44100, factor=20.):
    """
    Scale frequency axis logarithmically.
    """
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    scale = scale.astype(int)

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

def stft_matrix(audiopath, binsize=2**10, png_name='tmp.png',
                save_png=False, offset=0):
    """
    A function that converts a wav file into a spectrogram represented by a \
    matrix where rows represent frequency bins, columns represent time, and \
    the values of the matrix represent the decibel intensity. A matrix of \
    this form can be passed as input to the CNN after undergoing normalization.
    """
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    ims = np.flipud(ims)  

    return ims


@app.route("/audio", methods=['GET','POST'])
def audio_procesing():
    ans = "nothing"
    if request.method == "POST":
        audio_file = request.files['file']
        mat = stft_matrix(audio_file)
        features = np.mean(mat,axis=1)
        x = [features]
        x = sc.fit_transform(x)
        print(x.shape)
        output = audio_model.predict(x)
        print(output[0][0])
        if (output[0][0] >= 0.5):
            ans = "Depressed"
        else:
            ans = "Not Depressed"
        # res={{output[0][0]>=0.5?"depressed":"not depressed"}}
    return render_template('index.html', res = ans)

if __name__ == "__main__":
    app.run(port=8080, debug=True)