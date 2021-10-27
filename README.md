# wavemo
End-to-end emotion recognition from raw audio

**recognize emotion on your own voice**

A Convolutional Neural Network for Speech Emotion Recognition (SER) trained on the EMODB: http://emodb.bilderbar.info/download/.

`train.py` trains a model from scratch (400 epochs take ~17mins in an RTX GPU)
and saves best accuracy `.pth`.

Alternatively you can download a pretrained model (76,7% SER accuracy) from here https://drive.google.com/file/d/11h4XOqM2pvaPj9MhVCwGJmXKqWZUswYE/view?usp=sharing

`demo.ipynb` shows how to predict emotion on your own voice recording.

For `train.py` you should specify `cfg.data_dir` in `config.py` to point in the EMODB data.
For `demo.ipynb` you should specify `cfg.pth` in `config.py`
