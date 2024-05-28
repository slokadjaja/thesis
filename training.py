import torch
from torch.utils.data import DataLoader
from dataset import UCRDataset
from utils import get_ts_length, cat_kl_div, reconstruction_loss
from model import VAE
import matplotlib.pyplot as plt

dataset = "ArrowHead"

# Define constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = get_ts_length(dataset)
NUM_EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-2

# Load dataset
train = UCRDataset(dataset, "train")
test = UCRDataset(dataset, "test")
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)   # shape of batch -> [batch_size, 1, length]
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

# Define and train model
model = VAE(INPUT_DIM).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

loss_arr = []

for epoch in range(NUM_EPOCHS):
    for x, y in train_dataloader:
        x = x.to(device)

        logits, output = model(x)
        loss = reconstruction_loss(torch.squeeze(x, dim=1), output) + cat_kl_div(logits, 10, 2)
        loss.backward()
        optimizer.step()

    loss_arr.append(loss.item())

# Plot batch
# for i in range(4):
#     plt.plot(x[i].squeeze())

fig, ax = plt.subplots()
ax.plot(list(range(NUM_EPOCHS)), loss_arr)
ax.set(xlabel='Epoch', ylabel='Loss')

plt.show()
