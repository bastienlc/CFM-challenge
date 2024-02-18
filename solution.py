import numpy as np
import torch

from src.load import get_data_loaders, get_test_loader, load_data
from src.models import Base
from src.utils import predict, save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
X, y, X_test = load_data()
train_loader, _ = get_data_loaders(X, y, X, y, device, batch_size=128, shuffle=False)
test_loader = get_test_loader(X_test, device, batch_size=128)

embed_features = [0, 2, 3, 9]  # venue, action, side, trade
num_embed_features = [len(np.unique(X[:, :, i])) for i in embed_features]
encode_features = [1]

# Model
model = Base(
    X.shape[-1],
    24,
    embed_features,
    num_embed_features,
    encode_features,
    d_hidden=128,
    embedding_dim=8,
    num_layers=2,
).to(device)

load = "runs/18-02_18:57:47"

model.load_state_dict(torch.load(f"{load}/model.pt"))

# Test prediction
y_pred = predict(model, test_loader)
save(y_pred, "solution.csv")

# Score on full training set
y_pred = predict(model, train_loader)
print("Train accuracy:", (y_pred == y).mean())
