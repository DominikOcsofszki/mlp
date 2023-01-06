###########
# From: https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/blob/main/Variational_Inference_with_Normalizing_Flows/Flows.py
# https://www.youtube.com/watch?v=OiwtJA9In1U
###########
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

from MyDataSet import MyDataSets_Subset_4_9
from model import VaeFinal_only_one_hidden

torch.manual_seed(0)


def gaussian_log_pdf(z):
    """
    Arguments:
    ----------
        - z: a batch of m data points (size: m x data_dim)
    """
    return -.5 * (torch.log(torch.tensor([math.pi * 2], device=z.device)) + z ** 2).sum(1)


class PlanarFlow(nn.Module):

    def __init__(self, data_dim):
        super().__init__()

        self.u = nn.Parameter(torch.rand(data_dim))
        self.w = nn.Parameter(torch.rand(data_dim))
        self.b = nn.Parameter(torch.rand(1))
        self.h = nn.Tanh()
        self.h_prime = lambda z: (1 - self.h(z) ** 2)

    def constrained_u(self):
        """
        Constrain the parameters u to ensure invertibility
        """
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))

    def forward(self, z):
        u = self.constrained_u()
        hidden_units = torch.matmul(self.w.T, z.T) + self.b
        x = z + u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)
        log_det = torch.log((1 + torch.matmul(u.T, psi)).abs() + 1e-15)
        return x, log_det


class NormalizingFlow(nn.Module):

    def __init__(self, flow_length, data_dim):
        super().__init__()

        self.layers = nn.Sequential(
            *(PlanarFlow(data_dim) for _ in range(flow_length)))

    def forward(self, z):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians


def train(flow, optimizer, nb_epochs, log_density, batch_size, data_dim):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        # Generate new samples from the flow
        z0 = torch.randn(batch_size, data_dim).to(device)
        zk, log_jacobian = flow(z0)

        # Evaluate the exact and approximated densities
        flow_log_density = gaussian_log_pdf(z0) - log_jacobian
        exact_log_density = log_density(zk).to(device)

        # Compute the loss
        reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
        optimizer.zero_grad()
        loss = reverse_kl_divergence
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
    return training_loss


# def train_me(flow, optimizer, nb_epochs, log_density, batch_size, data_dim):
#     training_loss = []
#     for epoch in tqdm(range(nb_epochs)):
#         # Generate new samples from the flow
#         z0 = torch.randn(batch_size, data_dim).to(device)
def train_me(flow, optimizer, nb_epochs, log_density, batch_size, data_dim, MYDATASETS_SUBSET_4_9, model_latent_is_2):
    trainloader = MYDATASETS_SUBSET_4_9.dataloader_train_subset()
    testloader = MYDATASETS_SUBSET_4_9.dataloader_test_subset()

    # for epoch in range(nb_epochs):
    #     loop = tqdm(enumerate(trainloader))
    #
    #     for i, (x, _) in loop:

    training_loss = []
    # for epoch in tqdm(range(nb_epochs)):
    #     for X,_ in next(DataLoader_me):
    for epoch in range(nb_epochs):
        loop = tqdm(enumerate(trainloader))

        for i, (X, _) in loop:
            # for X,_ in mydatasets_subset_4_9.dataloader_test_subset():
            X = X.to(device)
            # Generate new samples from the flow
            # z0 = torch.randn(batch_size, data_dim).to(device)
            x, mu, sigma = model_latent_is_2(X)
            z0 = mu + sigma
            z0 = z0.to(device)
            zk, log_jacobian = flow(z0)

            # Evaluate the exact and approximated densities
            flow_log_density = gaussian_log_pdf(z0) - log_jacobian
            exact_log_density = log_density(zk).to(device)

            # Compute the loss
            reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
            optimizer.zero_grad()
            loss = reverse_kl_divergence
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
    return training_loss


def train_me_2(flow, optimizer, nb_epochs, log_density, trainloader):
    training_loss = []
    for epoch in range(nb_epochs):
        loop = tqdm(enumerate(trainloader))

        for i, (Z, _) in loop:
            z0 = Z.to(device)
            zk, log_jacobian = flow(z0)

            # Evaluate the exact and approximated densities
            flow_log_density = gaussian_log_pdf(z0) - log_jacobian
            exact_log_density = log_density(zk).to(device)

            # Compute the loss
            reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
            optimizer.zero_grad()
            loss = reverse_kl_divergence
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
    return training_loss


def plot_flow_density(flow, ax, lims=np.array([[-4, 4], [-4, 4]]), cmap="coolwarm", title=None,
                      nb_point_per_dimension=1000):
    # Sample broadly from the latent space
    latent_space_boundaries = np.array([[-15, 15], [-15, 15]]);
    xx, yy = np.meshgrid(
        np.linspace(latent_space_boundaries[0][0], latent_space_boundaries[0][1], nb_point_per_dimension),
        np.linspace(latent_space_boundaries[1][0], latent_space_boundaries[1][1], nb_point_per_dimension))
    z = torch.tensor(np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1), dtype=torch.float)
    # Generate data points and evaluate their densities
    zk, log_jacobian = flow(z.to(device))
    final_log_prob = gaussian_log_pdf(z) - log_jacobian.cpu()
    qk = torch.exp(final_log_prob)

    ax.set_xlim(lims[0][0], lims[0][1]);
    ax.set_ylim(lims[1][0], lims[1][1])
    ax.pcolormesh(
        zk[:, 0].detach().data.cpu().reshape(nb_point_per_dimension, nb_point_per_dimension),
        zk[:, 1].detach().data.cpu().reshape(nb_point_per_dimension, nb_point_per_dimension) * -1,
        qk.detach().data.reshape(nb_point_per_dimension, nb_point_per_dimension),
        cmap=cmap,
        rasterized=True,
    )
    if title is not None:
        plt.title(title, fontsize=22)


def plot_exact_density(ax, exact_log_density, lims=np.array([[-4, 4], [-4, 4]]), nb_point_per_dimension=100,
                       cmap="coolwarm", title=None):
    xx, yy = np.meshgrid(np.linspace(lims[0][0], lims[0][1], nb_point_per_dimension),
                         np.linspace(lims[1][0], lims[1][1], nb_point_per_dimension))
    z = torch.tensor(np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1))
    density = torch.exp(exact_log_density(z)).reshape(nb_point_per_dimension, nb_point_per_dimension)
    ax.imshow(density, extent=([lims[0][0], lims[0][1], lims[1][0], lims[1][1]]), cmap=cmap)
    if title is not None:
        plt.title(title, fontsize=22)


w1 = lambda z: torch.sin((2 * np.pi * z[:, 0]) / 4)
w2 = lambda z: 3 * torch.exp(-(((z[:, 0] - 1) / 0.6) ** 2) / 2)
w3 = lambda z: 3 * 1 / (1 + torch.exp(- ((z[:, 0] - 1) / 0.3)))
# Energy functions
U1 = lambda z: (((torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2) - 2) / 0.4) ** 2) / 2 - torch.log(
    1e-15 + (torch.exp(-(((z[:, 0] - 2) / 0.6) ** 2) / 2) + torch.exp(-(((z[:, 0] + 2) / 0.6) ** 2) / 2)))
U2 = lambda z: (((z[:, 1] - w1(z)) / 0.4) ** 2) / 2
U3 = lambda z: - torch.log(1e-15 + torch.exp(-(((z[:, 1] - w1(z)) / 0.35) ** 2) / 2) + torch.exp(
    -(((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2) / 2))
U4 = lambda z: - torch.log(1e-15 + torch.exp(-(((z[:, 1] - w1(z)) / 0.4) ** 2) / 2) + torch.exp(
    -(((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2) / 2))

if __name__ == "__main__":
    device = 'cpu'
    data_dim = 2
    index = 1

    ##PREPING DATA and loading model
    mydatasets_subset_4_9 = MyDataSets_Subset_4_9(batch_size_train=2)
    dataloader_testset_full = mydatasets_subset_4_9.dataloader_test_subset()
    dataloader_trainset_full = mydatasets_subset_4_9.dataloader_train_subset()
    # model = VaeFinal_only_one_hidden()  # Not trained yet
    import helper  # model_loaded = model.VaeMe_500_hidden()
    import model
    import torch

    pick_model = model.VaeFinal_only_one_hidden()
    model = helper.import_model_name(model_x=pick_model, activate_eval=True)
    # helper.show_scatter(examples=10000, model_loaded=pick_model, lat=2, in_train_class=False)

    # rand_tensor = torch.rand(2)
    # helper.latent_rand_to_img(pick_model, rand_tensor)

    #############################
    #############################

    import MyDataSet
    import pandas as pd
    import torch
    import model

    model_copy = model.VaeFinal_only_one_hidden_copy()
    model_copy = helper.import_model_name_weights_copy(model_x=model_copy, activate_eval=True)
    # dataset_test_4 = MyDataSet.MyDataSets_Subset_4(batch_size_train=-1)
    with torch.no_grad():
        dataset_49 = MyDataSet.MyDataSets_Subset_4_9(batch_size_train=-1)
        img_49_batch, label_49_batch = next(iter(dataset_49.train_loader_subset_changed_labels))
        rec49, mu49, sigma49 = model_copy(img_49_batch.clone())
        z = mu49
    df = pd.DataFrame({'z0': rec49[:, 0], 'z1': rec49[:, 1], 'labels': label_49_batch})
    mcd = MyDataSet.MyCustomDataset(df)
    BATCH_SIZE = 128
    dataloader = torch.utils.data.DataLoader(dataset=mcd, batch_size=BATCH_SIZE)
    #############################
    #############################

    plt.figure(figsize=(12, 12))
    # for U in [U1, U2, U3, U4]:
    for U in [U1]:
        # for U in [U1]:
        exact_log_density = lambda z: - U(z)

        # Plot the exact density
        ax = plt.subplot(5, 5, index);
        plt.xticks([], []);
        plt.yticks([], [])
        plot_exact_density(ax, exact_log_density, title=r'$\exp^{-U(z)}$' if index == 1 else None)
        index += 2

        # for flow_length in [32, 1, 1]:
        for flow_length in [100]:
        # for flow_length in [1, 2, 4]:
            flow = NormalizingFlow(flow_length, data_dim).to(device)
            optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
            # loss = train(flow, optimizer, 20000, exact_log_density, 4096, data_dim)
            # loss = train_me(flow, optimizer, 1, exact_log_density, 4096, data_dim,
            #                 MYDATASETS_SUBSET_4_9=mydatasets_subset_4_9, model_latent_is_2=model)
            loss = train_me_2(flow, optimizer, nb_epochs=50, log_density=exact_log_density,
                              trainloader=dataloader)
            # print(f'{loss = }')
            # train_me_2(flow, optimizer, nb_epochs, log_density, trainloader):

            # def train(flow, optimizer, nb_epochs, log_density, batch_size, data_dim):

            # Plot the learned density
            ax = plt.subplot(5, 5, index)
            plt.xticks([], [])
            plt.yticks([], [])
            plot_flow_density(flow, ax, title=f'K={flow_length}' if index <= 5 else None)
            index += 1
    # plt.savefig('Imgs/learned_densities.pdf')
    # plt.show()
    count = 0
    plt.savefig(f'NF_imgs/learned_densities_count_{count}.pdf')
    plt.show()
