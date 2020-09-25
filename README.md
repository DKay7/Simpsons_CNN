# Simpsons_CNN
>Сверточная нейронная сеть для решения задачи классификации симпсонов

>Датасет для данной сети можно взять из [конкуса на kaggle](https://www.kaggle.com/c/journey-springfield/submissions)

# Краткий обзор


---
## Вступление и важная информация о работе

Обзор предназначен для того, чтобы оградить читателя от необходимости разбираться в дебрях  непрофессионального кода автора. Он содержит в себе всю информацию о работе, графики, описание архатектур использованных моделей и результаты работы.

Тем не менее, для особенно смелых читателей, полный код работы представлен сразу же после обзора.

Прошу иметь в виду, что работа писалась на локальной машине (Windows), а на коллабе возникает конфликт библиотек или их версий (предположительно `sklearn`, `torch` и `pillow`), которого не было локально и который автору не очень хочется фиксить.

Конфигурация интерпретатора, которым пользовался автор, доступна [вот здесь](https://i.imgur.com/Qs8AW44.png)

---
## О моделях и балансировке классов

Лучшая точность на _несбалансированных_ данных была достигнута моделью из 5-и сверточных и 2-х полносвязных слоев с батч-нормализацией и составляла `93.836%`. Остальные модели, в том числе feature tuning последнего сверточного слоя `ResNet50` давали точность ниже, поэтому было решено проанализировать данные обучающей выборки. В ходе анализа было обнаружено, что количество изображений в классах сильно отличается - вплоть до того, что в одном из классов было `2246 изображений`, а в другом - всего `3 изображения`.


После того, как классы были сбалансированы, точность моделей сильно повысилась: теперь модель, состоящая из 6-и сверточных и 4-х полносвязных слоев, с батч-нормализацией на этапах свертки и классификации показала наилучшую точность в `99,468%`. Кроме того, в качестве эксперимента автор попробовал feature tuning последнего - 4-го - слоя (так называемый layer-4, содержащий несколько сверток) в моделях `Resnet-18` и `ResNet-50`, их точность была, соответственно, `96,705%` и `98,618%`.

---
## Модели

#### Первая модель
```python
class Cnn(nn.Module):
  
    def __init__(self, n_classes=42):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc1 = nn.Sequential(
          nn.Linear(96 * 5 * 5, 512)
          nn.BatchNorm1d(2048)
          nn.ReLU()
        )

        self.out = nn.Linear(512, n_classes)
  
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        logits = self.out(x)

        return logits
```

#### Модель, показавшая наилучшую точность по F1-мере

```python
class Cnn(nn.Module):
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
```
---
## Графики

Ниже представлены графики обучения для трех моделей, причем на графике авторской модели очень хорошо видно, как она начала учиться на несбалансированных классах, затем, _**на 27 эпохе**_, обучение было приостановлено, _**добавлена балансировка класов и аугментация всей обучеющей выборки**_, после чего обучение было запущено с той же эпохи и с теми же весами модели (обратите внимание, веса модели были теми же, что и во время приостановки обучения, а скачок loss вызван аугментированными данными). _**На 100 эпохе была убрана почти вся аугментация входных данных для валидационной части тренировочной выборки**_ (из трех преобразований осталось только зеркальное отражение относительно вертикальной оси). Это показалось хорошей идеей, ведь тогда точность на валидационной выборке станет приближена к точности на тестовой выборке.

Кроме того, видно, что благодаря аугментации и балансировке классов, loss валидационной части почти все время меньше, чем loss тренировочной части, что в сосвокупности с методом предотвращения переобучения, описанным ниже, гарантирует, что модель не переобучится.

---

>[Авторская модель](https://i.imgur.com/fN7R4Np.png)
>![Авторская модель](https://i.imgur.com/fN7R4Np.png) 

>[Feature tuning ResNet-50](https://i.imgur.com/kTp28Xz.png)
>![Feature tuning ResNet-50](https://i.imgur.com/kTp28Xz.png)

>[Feature tuning ResNet-18](https://i.imgur.com/zas12qd.png)
>![Feature tuning ResNet-18](https://i.imgur.com/zas12qd.png)

---

## Защита от переобучения

Хотелось бы отметить, что также была добавлена защита от переобучения. В конце каждой эпохи алгоритм сравнивает точность модели на валидационной части тренировочного датасета с ее лучшей точностью. Если точность модели оказывалась больше, чем ее лучшая точность, то обучаемые параметры модели сохранялись. После окончания обучения в модель загружались веса, при которых она давала лучшую точность на валидационной части выборки.

---

## Оптимизатор

В качестве оптимизатора спользовался алгоритм `AdamW`, со стандартным  `learning rate=1e-3`. Проводились эксперементы с модификацией `AMSGrad`, но значительных отличий выявить не получилось (тем не менее, была выбрана именно модификация `AMSGrad`, поскольку, согласно работам, которые приведены в документации `pyTorch`, она ускоряет сходимость).

Как шедулер `learning rate`'а использовался `ReduceLROnPlateau` с параметрами `factor=0.1`, `patience=2`.

---

## Заключение

В обзоре представлены только модели, показавшие наилучшую точность. Кроме них были опробованы `vgg16`, `inception v3`, `resnet-152`, и некоторые другие.

Все же, на взгляд автора, главным решением, повысившим точность моделей стала именно балансировка классов и аугментация обучающей выборки, ведь ни одна нейросеть не научиться определять класс по трем картинкам.

Хочется отметить, что автору было очень приятно, когда его модель показала точность лучшую, чем предобученные `ResNet`'ы, но объяснить себе автор это так и не смог. Изначально предположение состояло в том, что `ResNet`'ы просто глубже, и поэтому сходятся медленнее, но оно разбилось о графики, ведь из графиков четко видно, что модель вышла на "плато" и ее loss не изменялся значительно в течение достаточного числа эпох.

Полный код проекта доступен на [GitHub](https://github.com/DKay7/Simpsons_CNN).



