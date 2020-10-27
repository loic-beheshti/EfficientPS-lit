import torch
from modules.efficientPS import EfficientPS

model = EfficientPS()
model.forward(torch.rand(4, 3, 1024, 2048))