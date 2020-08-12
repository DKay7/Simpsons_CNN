from Network import Cnn, NetworkStuff
from DataBalancing import Regularisation

import torchvision
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn


regularisation = Regularisation()
# regularisation.regularise()


torch.cuda.empty_cache()
model = Cnn().to(torch.device("cuda"))

loss_func = nn.CrossEntropyLoss()
optimiser = optim.AdamW(model.parameters(), amsgrad=True)
lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.1, patience=2)


net_stuff = NetworkStuff(model,
                         optimiser,
                         loss_func,
                         lr_scheduler,
                         train_dir='data/train/simpsons_dataset',
                         test_dir='data/test/testset',
                         save_name='model8_adamW',
                         use_scheduler=True,
                         epochs=100000,
                         batch_size=336

                         )

net_stuff.load_model()
# net_stuff.train(clear_history=False)

net_stuff.submit()
# net_stuff.plotter()

net_stuff.draw_prediction(type_='val')
