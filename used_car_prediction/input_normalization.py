from torch import tensor
from numpy import float32
import torch
import torch
import pandas as pd
from torch import nn
import sys

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
print(X)
sys.exit()

Y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))
print(Y)

y_mean = Y.mean()
y_std = Y.std()
Y = (Y - y_mean) / y_std


model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.00000000001)

for i in range(0, 1000):
    optimizer.zero_grad()
    ouputs = model(X)
    loss = loss_fn(ouputs, Y)
    loss.backward()
    optimizer.step()

    # if i % 100 == 0:
    #     print(model.bias)
    #     print(model.weight)
prediction = model(torch.tensor([
    [5.0, 10000.0],
    [5.0, 10000.0]
], dtype=torch.float32))


print(prediction * y_std + y_mean)