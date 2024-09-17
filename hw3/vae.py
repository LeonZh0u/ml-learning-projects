import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Here, we use a simple MLP encoder and decoder, and parameterize the latent
        The encoder and decoder are both MLPs with 2 hidden layers, whose activation functions are all ReLU, i.e., 
        encoder: input_dim -> hidden_dim -> hidden_dim -> ??? (what is ??? here, as we need to output both mu and var?)
        decoder: latent_dim -> hidden_dim -> hidden_dim -> input_dim.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # TODO: instantiate the encoder and decoder.
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(), nn.Linear(self.hidden_dim, 2*self.latent_dim))
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.input_dim),nn.Sigmoid())
        
    def encode(self, x):
        """
        Probabilistic encoding of the input x to mu and sigma.
        Note: 
            sigma needs to be (i) diagnal and (ii) non-negative, 
            but Linear() layer doesn't give you that, so you need to transform it.
        Hint: 
            (i) modeling sigma in the form of var,
            (ii) use torch.log1p(torch.exp()) to ensure the non-negativity of var.
        """
        d = self.encoder(x)
        mu, var = d[:, :self.latent_dim], d[:, self.latent_dim:]
        return mu, torch.log1p(torch.exp(var))
    
    def reparameterize(self, mu, var):
        """
        Reparameterization trick, return the sampled latent variable z.
        Note: 
            var is the variance, sample with std.
        """
        return mu + torch.randn_like(mu) *torch.sqrt(var)
    
    def decode(self, z):
        """
        Generation with the decoder.
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        The forward function of the VAE.
        Returns: 
            (i) x_hat, the reconstructed input;
            (ii) mu, the mean of the latent variable;
            (iii) var, the variance of the latent variable.
        """
        mu, var = self.encode(x)
        return self.decode(self.reparameterize(mu, var)), mu, var
    
def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = torch.inverse(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = torch.trace(iS1 @ S0)
    det_term  = torch.log(torch.det(S1)/torch.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ torch.inverse(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    """

    # hyper-parameters setting.
    indim = 28 * 28
    hiddim = 100
    latentdim = 2
    epoch = 10
    batch_size = 16
    lr = 3e-3
    
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
    dataset = datasets.MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # model instantiation
    model = VAE(indim, hiddim, latentdim)

    # optimizer instantiation
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    BCEloss = nn.BCELoss()
    reconstruct_losses, kl_losses = [], []
    for ep in range(epoch):
        for x, _ in tqdm(train_loader):
            # flatten the x
            x = torch.flatten(x, start_dim=1)

            # foward of the x
            x_hat, mu, var = model(x)

            # calculate the reconstruction loss,
            # where we use binary cross entropy loss as the reconstruction loss, not the MSE.
            
            bce_loss = BCEloss(x_hat, x)

            # calculate the kl-divergence loss.
            kl_loss = kl_mvn(mu, var, torch.zeros_like(mu), torch.ones_like(var))

            # calculate the total loss as the sum of the reconstruction loss and the kl-divergence loss.
            loss = bce_loss+kl_loss
            
            # optimization of the model.
            loss.backward()
            optimizer.step()

            # log the losses.
            reconstruct_losses.append(bce_loss)
            kl_losses.append(kl_loss)
        
        print(f'epoch[{ep}]')
        print(f'losses | rec=[{reconstruct_losses[-1]:.4f}] | kl=[{kl_losses[-1]:.4f}]')

    # checkpoint the model.
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "vae_checkpoint")
    
    import matplotlib.pyplot as plt
    plt.plot(range(len(reconstruct_losses)), reconstruct_losses, 'g', label='Reconstruction Losses')
    # plt.plot(range(len(kl_losses)), kl_losses, 'b', label='KL-Div Losses')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'losses.png'))