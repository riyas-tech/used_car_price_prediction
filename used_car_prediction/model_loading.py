import torch
import sys
import pandas as pd
import torch
from torch import nn

X_mean = torch.load("./used_car_prediction/model/X_mean.pt", weights_only=True)
X_std = torch.load("./used_car_prediction/model/X_std.pt", weights_only=True)

Y_mean = torch.load("./used_car_prediction/model/Y_mean.pt", weights_only=True)
Y_std = torch.load("./used_car_prediction/model/Y_std.pt", weights_only=True)

model = nn.Linear(2, 1)
model.load_state_dict(
    torch.load("./used_car_prediction/model/model.pt", weights_only=True)
)
model.eval()
x_data = torch.tensor([[5,10000], [2, 10000], [5, 20000]], dtype=torch.float32)    

with torch.no_grad():
    prediction = model((x_data - X_mean) / X_std)
    print(prediction * Y_std + Y_mean)
