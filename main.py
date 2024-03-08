import torch

from src.models import DiffPool
from src.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_features = [0, 2, 3, 9]  # venue, action, side, trade
num_embed_features = [6, 3, 2, 2]
encode_features = [1]
model = DiffPool(
    num_features=9,
    num_class=24,
    embed_features=embed_features,
    num_embed_features=num_embed_features,
    embedding_dim=8,
    dropout=0.1,
).to(device)

# Train
batch_size = 128
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
)
