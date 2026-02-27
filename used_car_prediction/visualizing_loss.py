from torch import tensor
from numpy import float32
import torch
import torch
import pandas as pd
from torch import nn
import sys
import matplotlib.pyplot as plt

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


Y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))
y_mean = Y.mean()
y_std = Y.std()
Y = (Y - y_mean) / y_std


model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
losses = []
for i in range(0, 10000):
    optimizer.zero_grad()
    ouputs = model(X)
    loss = loss_fn(ouputs, Y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.show()

x_data = torch.tensor([[5,10000], [2, 10000], [5, 20000]], dtype=torch.float32)    

prediction = model((x_data - X_mean) / X_std)

print(prediction * y_std + y_mean)