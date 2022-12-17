import torch.nn as nn



class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_SIZE = 28*28
        HIDDEN_SIZE = 100
        LATENT_SIZE = 20
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        ##
        self.img_to_hidden = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.hidden_to_latent = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)

        # self.fc2 = nn.Linear(100, 10)
        self.latent_to_image = nn.Linear(LATENT_SIZE, INPUT_SIZE)

    def encoder(self, image):
        flatten = self.flatten(image)
        hidden = self.img_to_hidden(flatten)
        latent = self.hidden_to_latent(hidden)
        return latent

    def decoder(self, encoded_image):
        flatten = self.flatten(encoded_image)
        reconstructed_image = self.latent_to_image(flatten)
        return reconstructed_image

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # print(x.shape)
        return x
