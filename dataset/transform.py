from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform
from config import Config

resize = [transforms.Resize(Config.image_size)]
randaug = []
preprocess = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
if Config.use_randaug:
    randaug.append(rand_augment_transform(
        config_str='rand-m7-mstd0.5', 
        hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
    ))

train_transform = transforms.Compose(resize + randaug + preprocess)

test_transform = transforms.Compose([
    transforms.Resize(Config.image_size), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
