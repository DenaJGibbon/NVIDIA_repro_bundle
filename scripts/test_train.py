import torch
import time
from pathlib import Path

# Confirm GPU
assert torch.cuda.is_available(), "GPU not available"

device = "cuda"
print("Using device:", torch.cuda.get_device_name(0))

# Simple model
model = torch.nn.Linear(128, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Fake data
x = torch.randn(10000, 128, device=device)
y = torch.randn(10000, 1, device=device)

# Output directory
outdir = Path("outputs")
outdir.mkdir(exist_ok=True)

print("Starting test run...")
for epoch in range(5):
    optimizer.zero_grad()
    preds = model(x)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
    time.sleep(1)

# Save model
torch.save(model.state_dict(), outdir / "model.pt")
print("Saved model to outputs/model.pt")

