# wavemo

End-to-end speech emotion recognition (SER) from raw-audio.


WAVEMO is a Convolutional Network inspired from <a href="https://arxiv.org/abs/1609.03193">wav2letter</a> and written in PyTorch.

**Recognize emotion on your own voice**

`demo.ipynb` predicts the emotion in your own voice recording (16,000Hz .wav) using a pretrained model.

You can download a pretrained model here <a href="https://drive.google.com/file/d/11h4XOqM2pvaPj9MhVCwGJmXKqWZUswYE/view?usp=sharing">epoch_138_val_speaker_[15]_accuracy_0000,76786.pth</a> and specify **cfg.pth** in `config.py`.

**Train WAVEMO from scratch**

Alternatively you can run `train.py` to train the model from scratch (full training takes 1 hour in a small laptop GPU), no pretraining needed!

To run `train.py` you have to download <a href="http://emodb.bilderbar.info/download/">EMODB dataset</a>http://emodb.bilderbar.info/download/ (40MB) and specify **cfg.data_dir** in `config.py`.