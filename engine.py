import audformat
import audeer
import soundfile as sf
import pandas as pd
import os
import augly.audio as audaugs
import torch
import numpy as np
import sys
from torch import nn
from config import cfg

pd.options.display.float_format = '{:.4g}'.format
dev = cfg.dev


def loss_fn(logits, target):
    '''Should I use sigmoid instead of softmax if emotions coexist?'''
    return nn.functional.cross_entropy(logits, target=target)


def build_emodb():
    '''Returns list of tuples
    [ ('wav/03a04Lc.wav', emotion_id),
      ('wav/16b10Lb.wav', emotion_id)]
    '''

    #### https://audeering.github.io/audformat/emodb-example.html

    # Prepare functions for getting information from file names
    def parse_names(names, from_i, to_i, is_number=False, mapping=None):
        for name in names:
            key = name[from_i:to_i]
            if is_number:
                key = int(key)
            yield mapping[key] if mapping else key

    files = sorted(
        [os.path.join('wav', f) 
            for f in os.listdir(os.path.join(cfg.data_dir, 'wav'))]
    )
    names = [audeer.basename_wo_ext(f) for f in files]

    emotions = list(parse_names(names, from_i=5, to_i=6,
                    mapping=cfg.emotion_mapping))

    y = pd.read_csv(
        os.path.join(cfg.data_dir, 'erkennung.txt'),
        usecols=['Satz', 'erkannt'],
        index_col='Satz',
        delim_whitespace=True,
        encoding='Latin-1',
        decimal=',',
        converters={'Satz': lambda x: os.path.join('wav', x)},
        squeeze=True,
    )
    y = y.loc[files]
    y = y.replace(to_replace=u'\xa0', value='', regex=True)
    y = y.replace(to_replace=',', value='.', regex=True)
    confidences = y.astype('float').values
    language = audformat.utils.map_language('de')

    # speaker_mapping = {
    #     3: {'gender': male, 'age': 31, 'language': language},
    #     8: {'gender': female, 'age': 34, 'language': language},
    #     9: {'gender': female, 'age': 21, 'language': language},
    #     10: {'gender': male, 'age': 32, 'language': language},
    #     11: {'gender': male, 'age': 26, 'language': language},
    #     12: {'gender': male, 'age': 30, 'language': language},
    #     13: {'gender': female, 'age': 32, 'language': language},
    #     14: {'gender': female, 'age': 35, 'language': language},
    #     15: {'gender': male, 'age': 25, 'language': language},
    #     16: {'gender': female, 'age': 31, 'language': language},
    # }
    db = audformat.Database(
        name='emodb',
        source=None,  # http://emodb.bilderbar.info/download/download.zip'
        usage=audformat.define.Usage.UNRESTRICTED,
        languages=[language],
        description='EMODB',
        meta={
            'pdf': (
                'http://citeseerx.ist.psu.edu/viewdoc/'
                'download?doi=10.1.1.130.8506&rep=rep1&type=pdf'
            ),
        },
    )

    # Media
    db.media['microphone'] = audformat.Media(
        format='wav',
        sampling_rate=16000,
        channels=1,
    )

    # # Raters
    db.raters['gold'] = audformat.Rater()

    # Schemes
    db.schemes['emotion'] = audformat.Scheme(
        labels=[str(x) for x in cfg.emotion_mapping.values()],
        description='Six basic emotions and neutral.',
    )
    db.schemes['confidence'] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=1,
        description='Confidence of emotion ratings.',
    )

    # Tables
    index = audformat.filewise_index(files)

    db['emotion'] = audformat.Table(index)
    db['emotion']['emotion'] = audformat.Column(
        scheme_id='emotion',
        rater_id='gold',
    )
    db['emotion']['emotion'].set(emotions)
    db['emotion']['emotion.confidence'] = audformat.Column(
        scheme_id='confidence',
        rater_id='gold',
    )
    db['emotion']['emotion.confidence'].set(confidences / 100.0)

    # return a list of tuples [(file, emotion)]
    t = db['emotion'].get()
    return list(zip(t.index, t.values[:, 0]))


class EmoNet(nn.Module):

    def __init__(self):

        super().__init__()

        n = 250
        self.body = nn.Sequential(
            nn.Conv1d(1, n, n, stride=160, padding=45),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 48, stride=2, padding=23),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 7, stride=1, padding=3),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 7, stride=1, padding=3),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool1d(48, stride=2, padding=23),
            #
            nn.Conv1d(n, n, 7, stride=1, padding=3),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 7, stride=1, padding=3),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 7, stride=1, padding=3),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 7, stride=1, padding=3),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool1d(48, stride=2, padding=23),
            #
            nn.Conv1d(n, n, 7, stride=1, padding=3),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 32, stride=1, padding=16),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 1, stride=1, padding=0),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            nn.Conv1d(n, n, 1, stride=1, padding=0),
            nn.BatchNorm1d(n),
            nn.ReLU(inplace=True),
            # OUT
            nn.Conv1d(n, cfg.num_classes, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.body(x)
        x, _ = x.max(2)
        return x


class EmoDS():
    def __init__(self,
                 db=None,
                 split='train'):
        self.split = split
        self.sel_speakers = cfg.speaker_assign[split]
        # 'wav/16b10Lb.wav'
        filt = [(f, e) for f, e in db if int(f[4:6]) in self.sel_speakers]
        print(f'ds_{split} length={len(filt)}')
        self.wav = [cfg.data_dir + f for f, e in filt]
        self.label = [cfg.name2id[e] for f, e in filt]
        self.aug = audaugs.Compose([
            audaugs.OneOf([audaugs.AddBackgroundNoise(),  # rain under umbrella recording
                           audaugs.ToMono()]  # does nothing
                          ),
            audaugs.OneOf([
                audaugs.Reverb(),
                # audaugs.Harmonic(),  # sponge stuffed attenuated mic
                # audaugs.HighPassFilter(),  # dried voice
                # audaugs.LowPassFilter(),  # speech through old sponge telephone banana
                # audaugs.PitchShift(),  # holo metallic voice
                # audaugs.Percussive(),  # wobbliness underwater
                # audaugs.TimeStretch()  # speedup-reverb-ghost-holo
            ]),
            audaugs.ToMono()]
        )

    def __len__(self):
        return len(self.wav)

    def __getitem__(self, index):
        index = index % len(self)
        x, sample_rate = sf.read(self.wav[index])
        if self.split == 'train':
            # cut segment
            cut = np.random.randint(0, max(0, len(x) - 4000))  # let 4000 trailing samples
            x = np.concatenate([x[cut:cut + 4000],
                                x[:cut],
                                x[cut + 4000:]], 0)
            # cyclic rotation
            i = np.random.randint(0, len(x))
            x = np.roll(x, i)
            x, _ = self.aug(x,
                            sample_rate=sample_rate,
                            metadata=[])
            # crop 3s (equal durations for batch concatenation)
            x = x[:48000]
            x = np.pad(x,
                       (0, max(0, 48000 - len(x))))  # zeros if x<3s
        return x[None, :].astype(np.float32), self.label[index]

# import pygame
# from scipy.io.wavfile import write
# def play_np_array(x):
#     amplitude = np.iinfo(np.int16).max
#     data = amplitude * x
#     write('aa.wav', 16000, data.astype(np.int16))
#     pygame.mixer.init()
#     pygame.mixer.music.load('aa.wav')
#     pygame.mixer.music.play()


class Trainer():
    def __init__(self):

        db = build_emodb()  # create once

        self.ds_train = EmoDS(db=db, split='train')
        self.ds_val = EmoDS(db=db, split='val')

        self.dl_train = torch.utils.data.DataLoader(
            self.ds_train, 
            batch_size=16, shuffle=True, 
            num_workers=4, drop_last=True)

        self.dl_val = torch.utils.data.DataLoader(
            self.ds_val, 
            batch_size=1, shuffle=False, 
            num_workers=1, drop_last=False)

        self.model = EmoNet().to(dev)
        self.opt = torch.optim.AdamW(list(self.model.parameters()), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                        self.opt, [74], gamma=0.1)

        self.best_ser = 0
        self.last_pth = ''

    def fit(self):
        for epoch in range(cfg.num_epochs):
            print(f'__________\n EPOCH {epoch}')

            # train
            self.model.train()
            for b, (x, label) in enumerate(self.dl_train):
                self.opt.zero_grad()
                logits = self.model(x.to(dev))
                loss = loss_fn(logits, label.to(dev))
                loss.backward()
                self.opt.step()
                sys.stdout.flush()
                sys.stdout.write(f'loss={loss}\r')

            # validation
            self.model.eval()
            confusion = np.zeros((cfg.num_classes, cfg.num_classes))
            with torch.no_grad():
                for x, label in self.dl_val:
                    logits = self.model(x.to(dev))
                    pr_label = logits.argmax(1).cpu().numpy()
                    confusion[pr_label, label] += 1
            # print(pd.DataFrame(data=confusion,
            #                    index=cfg.emotions, 
            #                    columns=cfg.emotions)
            SER = np.diag(confusion).sum() / max(1, confusion.sum())
            print('SER accuracy=', SER)

            # save
            if SER > self.best_ser:
                acc_str = f'{SER:010.5f}'.replace('.', ',')
                pth = (f'{cfg.net_dir}_epoch_{epoch}'
                       f'_val_speaker_{self.ds_val.sel_speakers}'
                       f'_accuracy_{acc_str}.pth')
                print(f'SAVING NEW BEST MODEL: {pth}')
                torch.save(self.model.state_dict(), pth)
                try:
                    os.remove(self.last_pth)
                except OSError:
                    pass
                self.last_pth = pth
                self.best_ser = SER

            # update learning rate    
            self.lr_scheduler.step()
