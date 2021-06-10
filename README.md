# Music-Genre-Classification-using-RCNN

Music Genre Classification model which can classify 10 different genre of songs. For this project I have used GTZAN dataset.The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. 
The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

## Data Preprocessing
Deep neural networks require a large amount of input data to learn representative features. However, the GTZAN dataset
contain only 1000 song, which is insufficient for deep neural networks. To increase the number of training songs,
we divide each song into shorter 3 second music clips
with a 50% overlap. Then we apply the fast Fourier transform (FFT)
to frames of length 37500 and use the absolute value of each FFT frame.
We finally construct a STFT spectrogram with 128 frames,
and each frame is represented as a 513-dimensional vector.

Then we will split the dataset into **10% testing**, **10% validation** and **80% training**.
