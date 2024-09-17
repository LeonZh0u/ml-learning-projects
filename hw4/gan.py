import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
import os

def image_grid(images, n_rows):
    """Helper function to visualize a batch of images."""
    import matplotlib.pyplot as plt
    n_images = len(images)
    n_cols = n_images // n_rows
    if n_images % n_rows != 0: n_cols += 1
    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(image, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    return plt

class DNet(nn.Module):
    """
    We have implemented the Discriminator network for you.
    Note: 
        (i) The input image is flattened in the forward function to a vector of size 784;
        (ii) The output of the network is a single value, which is the logit of the input image being real, 
        which means you need to use the binary cross entropy loss with logits to train the discriminator.
    """
    def __init__(self, in_features, hiddim, out_features=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hiddim = hiddim

        # Discriminator will down-sample the input producing a binary output
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hiddim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=hiddim, out_features=hiddim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=hiddim, out_features=out_features),
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class GNet(nn.Module):
    """
    You need to implement the Generator network.
        The architecture of the Generator network should be the same as the Discriminator network, 
        with only one difference: the final output layer should have a Tanh activation function.
    """
    def __init__(self, in_features, hiddim, out_shape):
        super(GNet, self).__init__()
        out_features = np.prod(out_shape)
        self.out_features = out_features
        self.out_shape = out_shape
        self.in_features = in_features
        
        self.hiddim = hiddim

        # Implement self.net with nn.Sequential() method, shown as below:
        # self.net = nn.Sequential(...)
        self.net = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hiddim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.3),
                nn.Linear(in_features=hiddim, out_features=hiddim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.3),
                nn.Linear(in_features=hiddim, out_features=out_features),
                nn.Tanh(),
            )
        
    def forward(self, x):
        """Returns a batch of generated images of shape [batch_size, *self.out_shape]."""
        return self.net(x).view(-1, *self.out_shape)
    
if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    """

    # hyper-parameters setting.
    in_shape = (28, 28)
    indim = np.prod(in_shape)
    hiddim = 100
    latentdim = 16
    epoch = 50
    batch_size = 16

    # Tune the learning rate for the generator and the discriminator.
    # You should observe the loss of the discriminator to be close to 0.69=ln(2) at the beginning of the training.
    lr_g = 0.0002
    lr_d = 0.0002
    
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # preparing dataset
    # we transform the pixels of the image to be {-1, 1}, 
    # which is consistent with the output of the generator.
    tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, transform=tranform, download=True)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    # model instantiation
    # model_g = GNet(???)
    # model_d = DNet(???)
    model_g = GNet(latentdim, hiddim, in_shape)
    model_d = DNet(indim, hiddim, 1)

    # optimizer instantiation with both adam optimizer.
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr = lr_g)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr = lr_d)

    # fixed set of latents z, used to visualize the generated images
    # and compare the quality of the generated images.
    fixed_z = torch.randn((100, latentdim))
    criterion = nn.BCEWithLogitsLoss()

    losses_g, losses_d = [], []
    step = 0
    for ep in range(epoch):
        print(f'epoch[{ep}]')
        for x, _ in train_loader:
            # squeeze the x
            real_x = torch.squeeze(x, dim=1)

            # generate the fake x using the generator
            # latent vector z sampling from the normal distribution
            z = torch.normal(0, 1, size = (batch_size, latentdim))
            fake_x = model_g(z)

            # train the discriminator with binary cross entropy loss.
            # 1. concatenate the real_x and fake_x along the batch dimension.
            #    Note: you need to detach the fake_x from the computational graph, using .detach().clone()
            optimizer_d.zero_grad()
            logits = model_d(torch.cat((fake_x.detach().clone(), real_x),0))
            
            # 2. concatenate the real_y and fake_y along the batch dimension.
            
            # 3. compute the logits of the concatenated x.
            
            # 4. compute the binary cross entropy loss with logits
            loss_d = criterion(logits, torch.cat((torch.zeros(batch_size, 1), torch.ones(batch_size, 1)), 0))
            
            # 5. append the loss to losses_d
            losses_d.append(loss_d.detach().clone())
            loss_d.backward()
            # 6. update the discriminator for one step
            optimizer_d.step()

            # update the generator for one step
            # 1. compute the logits of the fake_x
            logits = model_d(fake_x)
            
            # 2. compute the binary cross entropy loss with logits, for only the fake_x
            loss_g = criterion(logits, torch.zeros(batch_size, 1))
            # 3. append the loss to losses_g
            losses_g.append(loss_g.detach().clone())
            
            # 4. update the generator for one step
            loss_g.backward()
            
            # log the losses.
            step += 1
            if step % 100 == 0:
                print(loss_d, loss_g)
                print(f'step {step} | loss_d=[{loss_d:.4f}] | loss_g=[{loss_g:.4f}]')
        
        # visualize the generated images of the fixed zs. 
        with torch.no_grad():
            fake_x = model_g(fixed_z) * 0.5 + 0.5
            fake_x = fake_x.numpy().reshape(-1, 28, 28)
            plt = image_grid(fake_x, n_rows=10)
            plt.suptitle(f'epoch {ep}')
            plt.savefig(os.path.join(log_dir, f'epoch_{ep}.png'))

    # checkpoint the model.
    torch.save(model_d.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pt'))
    torch.save(model_g.state_dict(), os.path.join(checkpoint_dir, 'generator.pt'))
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(range(len(losses_d)), losses_d, 'g', label='Discriminator Losses')
    plt.plot(range(len(losses_g)), losses_g, 'b', label='Generator Losses')
    plt.title('Training Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'losses.png'))