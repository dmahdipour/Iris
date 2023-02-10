import torch
import numpy as np
from myModel import irisModel
from dataSet import irisDataSet
from matplotlib import pyplot


epochs = 50
trainSetLen = 120
testSetLen = 150

# Dataset Name, Training_data percentage
trainDS = irisDataSet('Iris.csv', 80)
train_input = trainDS.train_data
train_target = trainDS.train_target
test_input = trainDS.test_data
test_target = trainDS.test_target

myModel = irisModel()

criteion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(myModel.parameters(), lr=0.1, momentum=0.9)
losses = []
for epoch in range(epochs):
    inputs = torch.autograd.Variable(torch.Tensor(train_input).float())
    targets = torch.autograd.Variable(torch.Tensor(train_target).long())
    optimizer.zero_grad()
    out = myModel(inputs)
    loss = criteion(out, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())


pyplot.plot(np.arange(len(losses)), losses)
pyplot.show()

tInput = torch.autograd.Variable(torch.Tensor(test_input).float())
tTarget = torch.autograd.Variable(torch.Tensor(test_target).float())

optimizer.zero_grad()
out = myModel(tInput)
_, predicted = torch.max(out.data, 1)

error_count = test_target.size - \
    np.count_nonzero((tTarget == predicted).numpy())

print("Accuracy rate is: ", (len(test_input)-error_count)/len(test_input)*100, '%')
