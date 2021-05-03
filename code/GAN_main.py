# -*- coding: utf-8 -*-
"""MetaGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10-4_2nuO4c_L8cfmDaiLps8CIXMwFOGg
"""

import math
import model
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np

class dataset(torch.utils.data.Dataset):    
    def __init__(self, data_path):
        self.data=np.load(data_path, allow_pickle=True)

        self.data[self.data=='setosa']=0.0
        self.data[self.data=='versicolor']=1.0
        self.data[self.data=='virginica']=2.0

        # self.data[:,-1].astype('int32')

    def __getitem__(self, index):
        x, y = torch.tensor(self.data[index,:-1].astype('float32')), torch.tensor(self.data[index,-1])

        return x,y

    def __len__(self):
        return len(self.data)

dataset=dataset('/content/iris.npy')

train_loader = torch.utils.data.DataLoader(
                                 dataset, batch_size=16,
                                 shuffle=True, num_workers=1,)

model2 = model.Generator().to('cuda')
model1 = model.Discriminator().to('cuda')
device='cuda'

#optimizer 

lr_init = 0.01
# optimizer1 = torch.optim.Adam(model1.parameters(), lr = lr)
# optimizer2 = torch.optim.Adam(model2.parameters(), lr = lr)

import matplotlib.pyplot as plt
for j in range(500):
     for i , (image , label) in enumerate(train_loader):
         
         lr = lr_init*(0.1)**(j//100)
         optimizer1 = torch.optim.Adam(model1.parameters(), lr = lr)
         optimizer2 = torch.optim.Adam(model2.parameters(), lr = lr)

         optimizer1.zero_grad()
         image.to('cuda')
         
         Dloss = model1.loss(image , model2)
         Dloss.backward()
         optimizer1.step()
         
         for k in range(3):
           optimizer2.zero_grad()
     
           Gloss , _  = model2.loss(image , model1)
           Gloss.backward()
           optimizer2.step()
           
           if i%10 == 0:
             print(j,Dloss)
             z = (torch.randn(1, 100 , 1 , 1 ).to('cpu'))
             output = model2(z)
            #  plt.imshow(output.cpu().detach().numpy().reshape(4,1)  , cmap='Greys_r')
             plt.show()

