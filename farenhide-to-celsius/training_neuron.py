import torch
from numpy import dtype
from numpy import float32
import torch
from torch import nn

# Input : Temperature in Celcius
x1 = torch.tensor([[10.0]], dtype=torch.float32)

# Acutal value : Temperature F
y1 = torch.tensor([[[50.0]]], dtype=torch.float32)

#Input : Temperature in Celcius
x2 = torch.tensor([[37.78]], dtype=torch.float32)

# Actual value : Temperature : F
y2 = torch.tensor([[100.0]], dtype=torch.float32)

# Model initialization , one input and one output
# input = Temperature in Celcius
# Output = Temperature in F

model = nn.Linear(1,1)

# Mention which loss funciton we need to use
# Mean Squared Error Loss Function 
loss_fn = torch.nn.MSELoss()

# How the model should be optimize 
# Stochastic Gradient Descent optimization algorithm, used to update 
# model parameters and minimize the loss function. 
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for i in range(0, 50000):
    #Tranining pass
    # First step : Tell the optimizer that we are going to do a new training 
    # zer_grad means - start from scratch 
    optimizer.zero_grad()

    # fetch the output for model 
    outputs = model(x1)

    #Calculate the loss function for first output 
    loss = loss_fn(outputs, y1)

    # Next step is , decrease the loss. Here it is calculating the gradient. 
    loss.backward()
    # After gradient calculation , tell optimizer 
    optimizer.step()

    #Second pass
    optimizer.zero_grad()
    outputs = model(x2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print("Bias   :", model.bias)
        print("Weight :", model.weight)


y1_pred = model(x1)
print("y1_pred :" ,y1_pred)