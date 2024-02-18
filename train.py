import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.deep import train
from src.load import load_data
from src.models import Base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
X, y, X_test = load_data()

embed_features = [0, 2, 3, 9]  # venue, action, side, trade
num_embed_features = [len(np.unique(X[:, :, i])) for i in embed_features]
encode_features = [1]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

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

# Train
batch_size = 128
epochs = 100
load = None

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

model = train(
    model,
    optimizer,
    scheduler,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=epochs,
    batch_size=batch_size,
    load=load,
)
