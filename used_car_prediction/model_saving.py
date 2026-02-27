from torch import tensor
from numpy import float32
import torch
import torch
import pandas as pd
from torch import nn
import sys
import os
if not os.path.isdir("./used_car_prediction/model"):
    os.mkdir("./used_car_prediction/model")


df = pd.read_csv("D:\\nlp_workrepo\\ml_learning\\used_car_prediction\\data\\used_cars.csv")

price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)


age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace("," , "")
milage = milage.str.replace(" mi.", "")
milage = milage.astype(int)
 

# Create Data as Tensors 
X= torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
]
)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

torch.save(X_mean, "./used_car_prediction/model/X_mean.pt")
torch.save(X_std, "./used_car_prediction/model/X_std.pt")

Y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))
y_mean = Y.mean()
y_std = Y.std()

Y = (Y - y_mean) / y_std
torch.save(y_mean, "./used_car_prediction/model/Y_mean.pt")
torch.save(y_std, "./used_car_prediction/model/Y_std.pt")


model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

for i in range(0, 10000):
    optimizer.zero_grad()
    ouputs = model(X)
    loss = loss_fn(ouputs, Y)
    loss.backward()
    optimizer.step()

    # if i % 100 == 0:
    #     print(loss)

x_data = torch.tensor([[5,10000], [2, 10000], [5, 20000]], dtype=torch.float32)    

prediction = model((x_data - X_mean) / X_std)

print(prediction * y_std + y_mean)