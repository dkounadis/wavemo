import os
import sys
import torch
from engine import EmoNet, EmoDS, build_emodb, benchmark, loss_fn
from pathlib import Path
from config import cfg

Path(cfg.net_dir).mkdir(parents=True, exist_ok=True)

db = build_emodb(cfg.data_dir)
ds_train = EmoDS(db=db, data_dir=cfg.data_dir, split='train')
ds_val = EmoDS(db=db, data_dir=cfg.data_dir, split='val')

dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True
    )

model = EmoNet().to(cfg.dev)
opt = torch.optim.AdamW(list(model.parameters()), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                opt, [74], gamma=0.1)

best_acc = 0
last_pth = ''

for ep in range(cfg.num_epochs):
    model.train()
    print(f'__________\nEPOCH {ep}')
    for b, (x, label) in enumerate(dl_train):
        opt.zero_grad()
        logits = model(x.to(cfg.dev))
        loss = loss_fn(logits, label.to(cfg.dev))
        loss.backward()
        opt.step()
        sys.stdout.flush()
        sys.stdout.write(f'loss={loss}\r')
    statistics = benchmark(model, ds=ds_val, dev=cfg.dev)
    acc = statistics['accuracy']
    print('SER accuracy=', acc)
    s = statistics['speaker']
    if acc > best_acc:
        pth = cfg.net_dir + f'epoch_{ep}_val_speaker_{s}_accuracy_{acc:010.5f}'.replace('.', ',')+'.pth'
        print(f'SAVING NEW BEST MODEL: {pth}')
        torch.save(model.state_dict(), pth)
        try:
            os.remove(last_pth)
        except OSError:
            pass
        last_pth = pth
        best_acc = acc
    lr_scheduler.step()
