import torch

from src.datasets import CFMGraphDataset
from src.loaders import get_test_loader, get_train_loaders
from src.models import GATEncoder
from src.utils import predict, save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader, val_loader = get_train_loaders(
    batch_size=128, shuffle=False, dataset=CFMGraphDataset
)
test_loader = get_test_loader(batch_size=128, dataset=CFMGraphDataset)
is_torch_geometric = True

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

load = "runs/night_test"

model.load_state_dict(torch.load(f"{load}/model.pt"))
model.eval()

# Test prediction
test_y_pred, _, test_probas = predict(
    model, test_loader, device, is_torch_geometric=is_torch_geometric
)
save(test_y_pred, "solution.csv")

# Score on validation set
val_y_pred, val_y_true, val_probas = predict(
    model, val_loader, device, is_torch_geometric=is_torch_geometric
)
print("Validation accuracy:", (val_y_pred == val_y_true).mean())

# Score on full training set
train_y_pred, train_y_true, train_probas = predict(
    model, train_loader, device, is_torch_geometric=is_torch_geometric
)
print("Train accuracy:", (train_y_pred == train_y_true).mean())

torch.save(test_probas, f"{load}/probas_test.pt")
torch.save(val_probas, f"{load}/probas_val.pt")
torch.save(train_probas, f"{load}/probas_train.pt")
