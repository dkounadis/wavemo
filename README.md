# wavemo

End-to-end speech emotion recognition (SER).


WAVEMO is an end-to-end ConvNet Architecture inspired from <a href="https://arxiv.org/abs/1609.03193">wav2letter</a> written in PyTorch. 
It is trained on <a href="http://emodb.bilderbar.info/download/">EMODB</a>'s raw audio signals without pretraining.

`train.py` trains a WAVEMO (400 epochs take ~17mins in an RTX GPU)
and saves best accuracy `.pth`. Alternatively you can download a pretrained checkpoint from here:
<td><a href="https://drive.google.com/file/d/11h4XOqM2pvaPj9MhVCwGJmXKqWZUswYE/view?usp=sharing">epoch_138_val_speaker_[15]_accuracy_0000,76786.pth</a></td>.
For `train.py` you should specify `cfg.data_dir` in `config.py` to point in the EMODB data.


**recognize emotion on your own voice**

`demo.ipynb` shows how to predict emotion on your own voice recording (16,000Hz .wav)

For `demo.ipynb` you should specify `cfg.pth` in `config.py` to point to the best_accuracy `.pth`.
