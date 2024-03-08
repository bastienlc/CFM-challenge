import torch

from src.datasets import CFMGraphDataset
from src.models import GATEncoder
from src.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GATEncoder(
    d_features=7,
    d_edges=2,
    d_out=24,
    d_hidden_dim=600,
    num_layers=5,
    num_heads=6,
    d_linear_layers=[256],
    dropout=0.01,
    activation="ReLU",
).to(device)

# Train
batch_size = 256
epochs = 100
load = None

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

model = train(
    model,
    optimizer,
    scheduler,
    epochs=epochs,
    batch_size=batch_size,
    load=load,
    dataset=CFMGraphDataset,
)
