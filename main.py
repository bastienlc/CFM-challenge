import torch

from src.datasets import CFMGraphDataset
from src.models import GATEncoder
from src.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_features = [0, 2, 3, 9]  # venue, action, side, trade
num_embed_features = [6, 3, 2, 2]
encode_features = [1]

model = GATEncoder(
    d_features=9,
    d_out=24,
    d_hidden_dim=120,
    num_layers=3,
    num_heads=3,
    d_linear_layers=[128],
    dropout=0.1,
    activation="ReLU",
).to(device)

# Train
batch_size = 512
epochs = 100
load = None

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
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
