from permissive_dict import PermissiveDict
import torch

cfg = PermissiveDict()
cfg.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.data_dir = 'emodb-src/'  # path to EMODB http://emodb.bilderbar.info/download/
cfg.net_dir = './networks/'

cfg.pth = 'epoch_138_val_speaker_[15]_accuracy_0000,76786.pth'
cfg.emotion_mapping = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'fear',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral',
}
cfg.emotions = list(cfg.emotion_mapping.values())
cfg.num_classes = len(cfg.emotions)
cfg.num_epochs = 400  # training 400 epochs takes 17min in RTX

# EMODB has recordings from 10 speakers.
# We use the recordings of 8 speakers as training set
#                       of 1 speaker  as val set
#                       of 1 speaker  as test set
cfg.speaker_assign = {
    # "all_emodb_speakers": [3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    "train": [3, 8, 9, 10, 11, 12, 13, 14],
    "val": [15],
    "test": [16]
}

cfg.name2id = dict(zip(cfg.emotions, range(cfg.num_classes)))
cfg.id2name = dict(zip(range(cfg.num_classes), cfg.emotions))
