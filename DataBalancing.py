from random import randint
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import pandas as pd

import time
import os


class Regularisation:
    def __init__(self,
                 directory='C:/Users/DKay/PycharmProjects/cnn/data/train/simpsons_dataset/'):

        self.dir_path = directory
        self.directory = os.listdir(path=os.path.join(self.dir_path))

        self.min_files = float('inf')
        self.max_files = float('-inf')

    @staticmethod
    def load_image(_file):
        image = Image.open(_file)
        image.load()

        return image, image.size

    def transform_image(self, file_):
        x, size = self.load_image(file_)

        transforms_train = transforms.Compose([
            transforms.RandomRotation(degrees=(-15, 15), expand=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=size),
        ])

        x = transforms_train(x)

        x.save(str(file_)[:-4] + str(randint(1770, 7000)) + ".jpg", "JPEG")

    def find_max_min(self):

        df = pd.DataFrame(columns=['character', 'total pics'])

        for index, folder in enumerate(self.directory):

            files_num = len(os.listdir(path=os.path.join(self.dir_path + folder)))

            df.loc[index] = {'character': folder, 'total pics': files_num}

            if files_num > self.max_files:
                self.max_files = files_num

            if files_num < self.min_files:
                self.min_files = files_num

        print(df)

        print('--------'
              '\nmax is {0}'
              '\nmin is {1}'.format(self.max_files, self.max_files))

        time.sleep(5)

    def regularise(self):
        self.find_max_min()

        if self.max_files != self.min_files:

            for folder in tqdm(self.directory):

                files_num = len(os.listdir(path=os.path.join(self.dir_path + folder)))
                files = os.listdir(path=os.path.join(self.dir_path + folder))

                while files_num < self.max_files:

                    for file in files:
                        files_num = len(os.listdir(path=os.path.join(self.dir_path + folder)))

                        if files_num >= self.max_files:
                            break

                        else:
                            path = os.path.join(self.dir_path + folder + '/' + file)
                            self.transform_image(path)
