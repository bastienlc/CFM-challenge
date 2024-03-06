import torch

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

# Model
model = Base(
    11,
    24,
    embed_features,
    num_embed_features,
    encode_features,
    d_hidden=128,
    embedding_dim=8,
    num_layers=2,
).to(device)

load = "runs/06-03_18:37:07"

model.load_state_dict(torch.load(f"{load}/model.pt"))

# Test prediction
y_pred = predict(model, test_loader)
save(y_pred, "solution.csv")

# Score on validation set
y_pred = predict(model, val_loader)
y = torch.zeros_like(y_pred)
for i, data in enumerate(val_loader):
    y[i * val_loader.batch_size : (i + 1) * val_loader.batch_size] = data[1]
print("Validation accuracy:", (y_pred == y).mean())

# Score on full training set
y_pred = predict(model, train_loader)
y = torch.zeros_like(y_pred)
for i, data in enumerate(train_loader):
    y[i * train_loader.batch_size : (i + 1) * train_loader.batch_size] = data[1]
print("Train accuracy:", (y_pred == y).mean())
