"""
This is an example to demonstrate DDP with multiple subgroups, using PyTorch's VAE example.
"""

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import torch.distributed as dist
import time

from utils import *


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, model, optimizer, train_loader, log_interval=10, group=None):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print0(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                ),
                process_group=group
            )
    print0(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        ),
        process_group=group
    )


def test(epoch, model, test_loader, batch_size, group=None):
    rank = dist.get_rank(group)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]]
                )
                os.makedirs(f"results-{rank}", exist_ok=True)
                save_image(
                    comparison.cpu(),
                    f"results-{rank}/reconstruction_" + str(epoch) + ".png",
                    nrow=n,
                )

    test_loss /= len(test_loader.dataset)
    print0("====> Test set loss: {:.4f}".format(test_loss), process_group=group)


def run(group_id, batch_size, epochs, group):
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank = dist.get_rank(group)
    local_size = dist.get_world_size(group)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, process_group=group)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if world_rank == 0:
        ## rank 0 download data first
        trainset = datasets.MNIST(
            "data", train=True, download=True, transform=transforms.ToTensor()
        )
        dist.barrier()
    else:
        ## others wait rank 0 to download data
        dist.barrier()
        trainset = datasets.MNIST(
            "data", train=True, download=True, transform=transforms.ToTensor()
        )

    sampler = torch.utils.data.distributed.DistributedSampler(trainset, rank=group_id, num_replicas=world_size//local_size)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=sampler
    )

    testset = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
    )
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer, train_loader, group=group)
        test(epoch, model, test_loader, batch_size=batch_size, group=group)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.module.decode(sample).cpu()
            os.makedirs(f"results-{rank}", exist_ok=True)
            save_image(
                sample.view(64, 1, 28, 28),
                f"results-{rank}/sample_" + str(epoch) + ".png",
            )

    dist.barrier()
    t1 = time.time()
    print(world_rank, "Done. time: %f" % (t1 - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument("--ngroups", type=int, help="number of groups", default=2)
    args = parser.parse_args()

    ngroups = args.ngroups
    comm_size, rank = setup_ddp()
    processes_groups = setup_ddp_groups(ngroups)

    for group_id, group in enumerate(processes_groups):
        if dist.get_rank(group) >= 0:
            run(group_id, args.batch_size, args.epochs + group_id, group=group)
