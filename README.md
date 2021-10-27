# wavemo

End-to-end speech emotion recognition (SER) from raw-audio.


WAVEMO is a Convolutional Net inspired from <a href="https://arxiv.org/abs/1609.03193">wav2letter</a> and written in PyTorch. It is trained on <a href="http://emodb.bilderbar.info/download/">EMODB</a> raw audio signals, no pretraining. `train.py` trains the model (400 epochs take ~17mins in an RTX GPU)
and saves best accuracy model (`.pth`). 
You should specify **cfg.data_dir** in `config.py` to point in the EMODB data.


**Recognize emotion on your own voice**

`demo.ipynb` predicts the emotion in your own voice recording (16,000Hz .wav).

You should specify **cfg.pth** in `config.py` to point to a pretrained model.
You can train your own model with `train.py` or download one here
<a href="https://drive.google.com/file/d/11h4XOqM2pvaPj9MhVCwGJmXKqWZUswYE/view?usp=sharing">epoch_138_val_speaker_[15]_accuracy_0000,76786.pth</a>. 