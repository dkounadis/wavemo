# wavemo
End-to-end emotion recognition from raw audio

A simple Convolutional Neural Network for emotion recognition using the Berlin Database of Emotional Speech EMODB.
You can download EMODB here http://emodb.bilderbar.info/download/

Run `train.py` (full train ~17mins in an RTX GPU) to train a model from scratch
or download a pretrained model (36MB) here https://drive.google.com/file/d/11h4XOqM2pvaPj9MhVCwGJmXKqWZUswYE/view?usp=sharing

Run `jupyter notebook test.ipynb` to predict emotion on your own voice recording (must be 16kHZ `.wav`)
