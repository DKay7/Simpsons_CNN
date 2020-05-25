import pickle
import torch
import numpy as np
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randint


class DatasetLoader(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(self, files, mode, rescale_size=96):
        super().__init__()
        self.data_modes = ['train', 'val', 'test']
        self.files = sorted(files)
        self.mode = mode
        self.rescale_size = rescale_size

        if self.mode not in self.data_modes:
            print(f"{self.mode} is not correct; correct modes: {self.data_modes}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('data/label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __getitem__(self, index):

        x, size = self.load_sample(self.files[index])

        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transforms_train = transforms.Compose([
            transforms.RandomRotation(degrees=(-7, 7), expand=True),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomResizedCrop(size=size),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transforms_val = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.mode == 'test':
            x = transforms_test(x)
            return x.to(torch.device("cuda:0"))

        elif self.mode == 'train':
            x = transforms_train(x)
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

        elif self.mode == 'val':
            x = transforms_val(x)
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()

        image = image.resize((self.rescale_size, self.rescale_size))

        return image, image.size
