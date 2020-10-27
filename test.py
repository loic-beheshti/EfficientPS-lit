import torch
from modules.efficientPS import EfficientPS

model = EfficientPS()
model.forward(torch.rand(1, 3, 1024, 2048))