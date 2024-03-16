from shutil import make_archive

import torch

from src.datasets import CFMGraphDataset
from src.models import GATEncoder
from src.train import train

saved = []
for k in range(100):
    try:
        torch.cuda.empty_cache()
        print(f"Training model {k + 1}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = GATEncoder(
            d_features=7,
            d_edges=5,
            d_out=24,
            d_hidden_dim=300,
            num_layers=3,
            num_heads=3,
            d_linear_layers=[256],
            dropout=0.01,
            activation="ReLU",
        ).to(device)

        batch_size = 256
        epochs = 20
        load = None

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        save_dir = train(
            model,
            optimizer,
            scheduler,
            epochs=epochs,
            batch_size=batch_size,
            load=load,
            dataset=CFMGraphDataset,
        )
        saved.append(save_dir)
    except Exception as e:
        print(e)
        continue

for dir in saved:
    make_archive(f"{dir}.zip", "zip", dir)
