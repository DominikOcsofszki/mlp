import torch
import torch.optim as optim
import torch.nn as nn



class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_SIZE = 28*28
        HIDDEN_SIZE = 100
        LATTENT_SIZE = 2
        self.relu = nn.ReLU()
        self.img_to_hidden = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.hidden_to_lattent = nn.Linear(HIDDEN_SIZE, LATTENT_SIZE)

        self.fc2 = nn.Linear(100, 10)

    def encoder(self, image):
        hidden = self.img_to_hidden(image)
        latent = self.hidden_to_lattent(hidden)
        return latent

    # def decoder
