import birdnet
import inspect
import torch

device = "GPU" if torch.cuda.is_available() else "CPU"

print("Device:", device)

print("\nload signature:")
print(inspect.signature(birdnet.load))

print("\nload_perch_v2 signature:")
print(inspect.signature(birdnet.load_perch_v2))

model = birdnet.load_perch_v2(device=device)

print("\nModel type:")
print(type(model))

print("\nModel dir:")
print([x for x in dir(model) if not x.startswith("_")])

print("\nEncode signature:")
print(inspect.signature(model.encode))
