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

## Model Training
For this project I have used parallel Bi-directional LSTM and CNN models. I have experimented many architectures and got some good validation accuracy results.

![Experiment Graphs](https://github.com/abdul789150/Music-Genre-Classification-using-RCNN/blob/main/images/accuracy_graph.png)

Final selected model configuration. 
CNN model Configuration is given below:
- 5 Conv2D layers.
- 2 Dropout layers.
- 4 MaxPooling layers.
- BatchNOrmalization layers are also used to speed up the computation of CNN model.

RNN model configuration is given below:
- 2 MaxPooling layer to reduce the dimensinality of spectogram into (128, 128) shape.
- 2 Bi-Directional LSTM layers

**In the end those 2 models will return 256 shape output and then I will just add those two models**

After combining the model I have just added one Dense layer with **10 neurons** and a **softmax function**.

## Results
Final Model **validation accuracy: 94%** and **training accuracy: 85.7%**.

![Model Result](https://github.com/abdul789150/Music-Genre-Classification-using-RCNN/blob/main/images/selected_model_graph.png)


### Confusion Matrix
![Confusion Matrix Result](https://github.com/abdul789150/Music-Genre-Classification-using-RCNN/blob/main/images/cm_rcnn.png)

## Libraries Required to run this project:
- numpy
- librosa
- scipy
- flask
- keras
- tensorflow
