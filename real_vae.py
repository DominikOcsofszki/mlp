import torch.nn as nn
import torch

#Input img -> Hidden dim -> mean,std -> Parametriziation trick -> decoder -> Output img
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200,z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim,h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear( z_dim,h_dim)
        self.hid_2img = nn.Linear( h_dim,input_dim)
        self.relu = nn.ReLU()

    def encode(self,x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.relu(self.hid_2mu(h)),self.relu(self.hid_2sigma(h))
        return mu,sigma


    def decode(self,z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self,x):
        mu, sigma = self.encode(x)
        eps = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*eps
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma

if __name__ == '__main__' :
    x = torch.randn(4,28*28)
    vae = VAE(input_dim=28*28)
    x_reconstructed,mu,sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)