import torch
import pickle
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from DataLoader import DatasetLoader
from matplotlib import pyplot as plt, ticker
from sklearn.metrics import f1_score


class Cnn(nn.Module):
    """Класс сверточной нейросети"""

    def __init__(self, n_classes=42):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(2048, n_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = f.interpolate(x, size=(4, 4), align_corners=False, mode='bilinear')
        x = x.view(x.size(0), 4 * 4 * 256)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


class NetworkStuff:
    """Класс функций для работы с нейросетью"""

    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 scheduler,
                 train_dir,
                 test_dir,
                 save_name,
                 use_scheduler=True,
                 epochs=25,
                 batch_size=64,
                 val_size=0.25):
        """

        :param model: нейросеть
        :param optimizer: оптимизатор
        :param criterion: функция ошибки
        :param scheduler: оптимизатор шага
        :param train_dir: директория тренировочной выборки
        :param test_dir: директория тестовой выборки
        :param save_name: имя для сохранеия файлов
        :param use_scheduler: использовать ли оптимизатор шага
        :param epochs: количество эпох
        :param batch_size: размер батча
        :param val_size: размер валидационной части обучающей выборки
        """

        self.device = torch.device("cuda")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.file_name = save_name
        self.use_scheduler = use_scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = val_size

        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.history = None

        self.get_dataset()

    def get_dataset(self):
        """
        Получает датасет из директории,
        переданной при создании объекта класса
        """

        train_val_files = sorted(list(self.train_dir.rglob('*.jpg')))
        test_files = sorted(list(self.test_dir.rglob('*.jpg')))

        train_val_labels = [path.parent.name for path in train_val_files]

        train_files, val_files = train_test_split(train_val_files,
                                                  test_size=self.test_size,
                                                  stratify=train_val_labels)

        self.val_dataset = DatasetLoader(val_files, mode='val')
        self.train_dataset = DatasetLoader(train_files, mode='train')
        self.test_dataset = DatasetLoader(test_files, mode="test")

    def fit_epoch(self, train_loader: torch.utils.data.DataLoader):
        """
        Реализует одну эпоху обучения


        :param train_loader: Загрузчик тренировочной части обучающей выборки

        :return: Ошибка и точность на тренировочной части обучающей выборки
            для построения графиков
        """

        running_loss = 0.0
        running_corrects = 0
        processed_data = 0

        for inputs, labels in tqdm(train_loader, position=0, leave=True):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            prediction = torch.argmax(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(prediction == labels.data)
            processed_data += inputs.size(0)

        train_loss = running_loss / processed_data
        train_acc = running_corrects.cpu().numpy() / processed_data

        return train_loss, train_acc

    def eval_epoch(self, val_loader: torch.utils.data.DataLoader):
        """
        Оценка модели по валидационной части

        :param val_loader: Загрузчик валидационной части обучающей выборки

        :return: Ошибка и точность на валидационной части обучающей выборки
            для построения графиков
        """

        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                prediction = torch.argmax(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(prediction == labels.data)
            processed_size += inputs.size(0)

        val_loss = running_loss / processed_size
        val_acc = running_corrects.double() / processed_size

        return val_loss, val_acc

    def train(self, file_name=None, clear_history=True):
        """
        Тренирует нейронную сеть
        
        :param file_name: Имя файла для истории обучения и весов модели, 
            если не передано, будет взято имя, переданное в конструктор класса
        
        :param clear_history: Очищать ли историю. Если False, то история
            обучения сохранится, чтобы вы могли построить график всего обучения,
            даже если останавливали его.

        """

        if file_name is None:
            file_name = self.file_name

        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False)
        if clear_history:
            self.history = []

        else:
            self.history = self.get_history()

        start_value = len(self.history)
        best_val_accuracy = 0.0
        best_weights = 0

        for epoch in tqdm(range(start_value, self.epochs+start_value), position=1,
                          leave=True, initial=start_value,
                          total=self.epochs+start_value):

            train_loss, train_acc = self.fit_epoch(train_loader)
            val_loss, val_acc = self.eval_epoch(val_loader)

            if self.use_scheduler:
                self.scheduler.step(val_loss)

            self.history.append((train_loss, train_acc, val_loss, val_acc))
            log_template = "\nEpoch {epoch:03d} train_loss: {train_loss:0.4f} \
                    val_loss {val_loss:0.4f} train_acc {train_acc:0.4f} val_acc {val_acc:0.4f}"
            tqdm.write(log_template.format(epoch=epoch+1,
                                           train_loss=train_loss,
                                           val_loss=val_loss,
                                           train_acc=train_acc,
                                           val_acc=val_acc))

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_weights = self.model.state_dict()
                self.save_model(file_name)
                self.save_history(file_name)

        self.model.load_state_dict(best_weights)

        print('Best val: {b_v:0.4f}'.format(b_v=best_val_accuracy*100))

        self.save_model(file_name)
        self.save_history(file_name)

    def predict(self, test_loader=None):
        """
        Возвращает предсказание нейронной сети.


        :param test_loader: Загрузчик тестовой выборки,
            если не передан, будет создан из директории,
            переданной в конструктор класса

        :return: Для каждой картинки возвращает
            вектор вероятностей ее принадлежности к
            одному из классов
        """

        if test_loader is None:
            test_loader = DataLoader(self.test_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)

        with torch.no_grad():
            logits = []

            for inputs in test_loader:
                inputs = inputs.to(self.device)
                self.model.eval()
                outputs = self.model(inputs).cpu()
                logits.append(outputs)

        probabilities = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()

        return probabilities

    def save_model(self, file_name=None):
        """
        Сохраняет параметры нейронной сети

        :param file_name: Имя файла для весов модели,
            если не передано, будет взято имя, переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '.pth'
        torch.save(self.model.state_dict(), path)

    def load_model(self, file_name=None):
        """
        Загружает параметры нейронной сети

        :param file_name: Имя файла для весов модели,
            если не передано, будет взято имя, переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '.pth'
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(torch.device('cuda'))

    def get_model(self):
        """
        Возвращает модель нейронной сети

        :return: модель нейронной сети
        """
        return self.model

    def submit(self, file_name=None):
        """
        Делает csv-сабмит и сохраняет его.

        :param file_name: Имя файла для сохранеия сабмита,
            если не передано, будет взято имя,
            переданное в конструктор класса
        """
        if file_name is None:
            file_name = self.file_name

        path = 'submits/' + file_name + '.csv'

        probabilities = self.predict()
        label_encoder = pickle.load(open("data/label_encoder.pkl", 'rb'))
        predictions = label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
        test_filenames = [path.name for path in self.test_dataset.files]

        df = pd.DataFrame()
        df['Id'] = test_filenames
        df['Expected'] = predictions
        df.to_csv(path, index=False)

    def plotter(self, history=None, file_name=None):
        """
        Строит графики точности и ошибки от эпохи обучения,
        затем сохраняет их под именем, переданным в конструктор
        класса

        :param file_name: Имя файла для сохранеия графика,
            если не передано, будет взято имя,
            переданное в конструктор класса

        :param history: история обучения, по которой
            нужно строить график. Если не передано, то
            будет использована история из директории
            /histories с именем файла, переданным в
            конструктор класса
        """

        if history is None:
            history = self.get_history()

        if file_name is None:
            file_name = self.file_name

        path = 'plots/' + file_name + '.png'

        loss, acc, val_loss, val_acc = zip(*history)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

        axes[0].plot(loss, label="train_loss")
        axes[0].plot(val_loss, label="val_loss")
        axes[0].legend(loc='best')
        axes[0].set_xlabel('iterations')
        axes[0].set_ylabel('loss')
        axes[0].set_xlim(left=0)

        axes[1].plot(acc, label="train_acc")
        axes[1].plot(val_acc, label="val_acc")
        axes[1].legend(loc='best')
        axes[1].set_xlabel('iterations')
        axes[1].set_ylabel('accuracy')
        axes[1].set_xlim(left=0)
        axes[1].set_ylim(top=1)

        axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.0625))
        axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))

        axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.015625))
        axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.0625))

        fig.suptitle(self.file_name)

        for ax in axes:

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            ax.tick_params(axis='x',
                           which='major',
                           labelsize=6,
                           direction='out',
                           labelrotation=77)

            ax.tick_params(axis='y',
                           which='major',
                           labelsize=8,
                           direction='out',
                           labelrotation=15)

            ax.tick_params(axis='both',
                           which='minor',
                           grid_alpha=0.3,
                           direction='out')

            ax.grid(which='major',
                    color='darkcyan')

            ax.minorticks_on()

            ax.grid(which='minor',
                    color='lightseagreen')

        fig.savefig(path)
        plt.show()

    def save_history(self, file_name=None):
        """
        Сохраняет историю обучения.

        :param file_name: Имя файла для сохранеия истории,
            если не передано, будет взято имя,
            переданное в конструктор класса
        """
        if file_name is None:
            file_name = self.file_name

        with open('histories/' + file_name + '.pickle', 'wb') as file:
            torch.save(self.history, file)

    def get_history(self, file_name=None):
        """

        :param file_name: Имя файла для загрузки истории,
            если не передано, будет взято имя,
            переданное в конструктор класса
        :return:
        """
        if file_name is None:
            file_name = self.file_name

        if self.history is None or self.history == []:

            with open('histories/' + file_name + '.pickle', 'rb') as file:
                self.history = torch.load(file)

        return self.history
