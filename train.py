import torch
import pandas as pd
from utils.preprocess import load_and_preprocess
from model.model import PriceMovementModel

# Load data
df = load_and_preprocess("data/btc.csv")

X = df[['return', 'ma_5', 'ma_10', 'volume_change']].values
y = df['target'].values

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Model
model = PriceMovementModel(input_size=X.shape[1])
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "model/btc_model.pt")
print("Model saved to model/btc_model.pt")

