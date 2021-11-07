import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

class FeedForward(nn.Module):
    def __init__(self, _in: int, _out: int):
        super().__init__()
        self.fc1 = nn.Linear(_in, _in//2)
        self.fc2 = nn.Linear(_in//2, _in//4)
        self.fc3 = nn.Linear(_in//4, _out)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.tensor(df.values).float().to(device)

def x_y_to_dataloader(x, y, batch_size, shuffle=True):
    data = data_utils.TensorDataset(df_to_tensor(x), df_to_tensor(y))
    dataloader = data_utils.DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader
    
def train(net, x, y, epochs, batch_size=32):
    trainloader = x_y_to_dataloader(x, y, batch_size, True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    
def eval(net, x, y, batch_size=32):
    testloader = x_y_to_dataloader(x, y, batch_size, False)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data
            # calculate outputs by running images through the network
            outputs = net(inputs)
            quant_outputs = torch.round(outputs*4)/4
            total += targets.size(0)
            correct += (quant_outputs == targets).sum().item()

    print(f'Accuracy of the network on the {len(testloader)*batch_size} test images: %d %%' % (
        100 * correct / total))