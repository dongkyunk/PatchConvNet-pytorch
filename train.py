from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer

from model.cifar100_model import Cifar100Model
from config import Config


seed_everything(Config.seed)
model = Cifar100Model(Config)
trainer = Trainer(gpus=1, max_epochs=Config.epochs, auto_lr_find=True)
# trainer.tune(model)
trainer.fit(model)