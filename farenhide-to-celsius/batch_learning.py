import torch
from numpy import dtype
from numpy import float32
import torch
from torch import nn

x = torch.tensor([
    [10.0],
    [37.78]
])
y = torch.tensor([
    [50.0],
    [100.0]

])

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
    outputs = model(x)

    #Calculate the loss function for first output 
    loss = loss_fn(outputs, y)

    # Next step is , decrease the loss. Here it is calculating the gradient. 
    loss.backward()
    # After gradient calculation , tell optimizer 
    optimizer.step()


    if i % 100 == 0:
        print("Bias   :", model.bias)
        print("Weight :", model.weight)


measurments = torch.tensor ([
    [37.5]
])

model.eval()
with torch.no_grad():
    prediction = model(measurments)
    print ("Prediction    :" , prediction)