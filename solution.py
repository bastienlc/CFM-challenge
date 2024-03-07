import numpy as np
import torch
from matplotlib import pyplot as plt

from src.load import get_test_loader, get_train_loaders
from src.models import Base
from src.utils import predict, save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader, val_loader = get_train_loaders(batch_size=128, shuffle=False)
test_loader = get_test_loader(batch_size=128)

embed_features = [0, 2, 3, 9]  # venue, action, side, trade
num_embed_features = [6, 3, 2, 2]
encode_features = [1]
model = Base(
    num_features=11,
    num_class=24,
    embed_features=embed_features,
    num_embed_features=num_embed_features,
    encode_features=encode_features,
    d_hidden=128,
    embedding_dim=8,
    num_layers=2,
    dropout=0.1,
).to(device)

load = "runs/07-03_17:12:33"

model.load_state_dict(torch.load(f"{load}/model.pt"))

# Test prediction
y_pred = predict(model, test_loader)
save(y_pred, "solution.csv")

# Score on validation set
y_pred = predict(model, val_loader)
y = np.zeros_like(y_pred)
for i, data in enumerate(val_loader):
    y[i * val_loader.batch_size : (i + 1) * val_loader.batch_size] = (
        data[1].cpu().numpy()
    )
print("Validation accuracy:", (y_pred == y).mean())
val_accuracies = []
for i in range(24):
    val_accuracies.append((y_pred[y == i] == i).mean())

# Score on full training set
y_pred = predict(model, train_loader)
y = np.zeros_like(y_pred)
for i, data in enumerate(train_loader):
    y[i * train_loader.batch_size : (i + 1) * train_loader.batch_size] = (
        data[1].cpu().numpy()
    )
print("Train accuracy:", (y_pred == y).mean())
train_accuracies = []
for i in range(24):
    train_accuracies.append((y_pred[y == i] == i).mean())

width = 0.2
plt.bar(np.arange(24) - width, val_accuracies, label="val", width=width)
plt.bar(np.arange(24) + width, train_accuracies, label="train", width=width)
plt.xticks(range(24))
plt.legend()
plt.show()
