from engine import Trainer
from pathlib import Path
from config import cfg

Path(cfg.net_dir).mkdir(parents=True, 
                        exist_ok=True)

trainer = Trainer()
trainer.fit()
