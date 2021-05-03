#GAN

import torch
from torch import nn, optim
from torchvision import datasets, transforms

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

device='cuda'

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator,self).__init__()  
        self.dense = nn.Linear(4, 1);
        self.activation = nn.Sigmoid()

    def forward(self,x):

        # dicrimination's loss
        x = x.to('cuda')

        D_output = self.activation(self.dense(x))

        return D_output

    def loss(self, x , generator_model):

        # dicrimination's loss
        # x = x.view(-1, 64*64)
        bs = len(x)
        x_real = x
        y_real = torch.ones( (bs,))
        y_real.type(torch.LongTensor)
        x_real, y_real = x_real.to(device), y_real.to(device)
        D_output = self.forward(x_real).view(-1)

        D_real_loss = criterion(D_output , y_real)

        # train discriminator on fake
        z = torch.randn(bs, 4).to('cuda')

        
        x_fake = generator_model(z) 
        y_fake = torch.zeros( (bs,))
        y_fake=y_fake.type(torch.LongTensor).to(device)
        
        D_output = self.forward(x_fake).view(-1)

        D_fake_loss = criterion(D_output, y_real*0)

        D_loss = D_real_loss + D_fake_loss  

        return D_loss
        

        

class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()
        self.dense_layer = nn.Linear(4, 4)
        self.activation = nn.Sigmoid()

        
    def forward(self , x):
        
        #GENERATOR LSS
        bs = len(x)
        z = torch.randn(bs, 4).to('cuda')
        G_output = self.activation(self.dense_layer(z))
        return G_output

    def loss(self , x , discriminator_model ):
        
        
        bs = len(x)
        z = torch.randn(bs, 100, 1 , 1 ,device=device)
        G_output = self.forward(z)

        y = torch.ones( (bs,))
        y.type(torch.LongTensor)
        y = y.to(device)
        
        D_output = discriminator_model(G_output).view(-1)

        G_acc = ((torch.round(D_output)==y).sum().item())*100
        # eps = 0.0000000000000000000001
        G_loss = criterion(D_output, y)

        return G_loss , G_acc

criterion = nn.BCEWithLogitsLoss()