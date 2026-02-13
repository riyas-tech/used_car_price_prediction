import torch
import torch
from torch import nn

x = torch.tensor([
    [10.0],
    [38.0],
    [100.0],
    [150.0]
])

model = nn.Linear(1, 1)

print(model)

print("Bias  :" , model.bias)
print("Weight : ", model.weight)

model.bias = nn.Parameter(
    torch.tensor([32.0])
)

model.weight = nn.Parameter(
    torch.tensor([[1.8]])
)
y_pred = model(x)
print(y_pred)