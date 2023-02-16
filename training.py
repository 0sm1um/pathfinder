import pandas as pd;
from scipy.stats import zscore
import torch as torch;
import numpy as np
import torchvision.datasets as datasets
from torchvision import models, transforms
import torch.nn as nn;
import torch.nn.functional as F;
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
np.random.seed(42)
# Use GPU if available, else use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('Device set to: '+str(device))

# TODO Figure out proper Normalization
#transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])

#transform = transforms.Compose(
#        [transforms.Resize((224, 224)),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])])

'''Data Pre Processing:'''

class SimulatedTrajectoryDataset(Dataset):
    """Simulated Trajectories PyTorch Dataset Class"""

    def __init__(self, tensor, transform=None):
        assert torch.is_tensor(tensor) == True
        self.trajectory_data = tensor
        self.transform = transform

    def __len__(self):
        return self.trajectory_data.size(dim=0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.trajectory_data[idx][0:18]
        model = self.trajectory_data[idx][19]
        if self.transform:
            sample = self.transform(sample)
        return sample, model

data_tensor = torch.load('full_data_tensor.pt',map_location = device)
constant_velocity_weight = (data_tensor[:,19] == 0.).sum(dim=0)/len(data_tensor)
right_turn_weight = (data_tensor[:,19] == 1.).sum(dim=0)/len(data_tensor)
left_turn_weight = (data_tensor[:,19] == 2.).sum(dim=0)/len(data_tensor)
weights = torch.tensor([constant_velocity_weight, right_turn_weight,left_turn_weight],device=device)

print('Class Weights: '+ str(weights))

normalized_data_tensor = nn.functional.normalize(data_tensor,p=1,dim=1)

predictions_dataset = SimulatedTrajectoryDataset(normalized_data_tensor)


train_dataset, test_dataset = random_split(predictions_dataset,
                                           [int(np.ceil(len(predictions_dataset)*0.7)),
                                            int(np.floor(len(predictions_dataset)*0.3))])
print('Partitioned into training and testing data of lengths: '+
      str(len(train_dataset))+' and '+str(len(test_dataset)))

sampler = WeightedRandomSampler(weights, num_samples = len(train_dataset), replacement = True)

batch_size=128;
#trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,sampler = sampler)
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

'''Here we define the neural network:'''
# create a neural network (inherit from nn.Module)

input_size = 18
num_classes = 3 # Num Classes
hidden_size = 50

class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer3(x)
        #return F.log_softmax(x, dim=1) # For NLL Loss
        return F.softmax(x, dim=1) # For Cross Entropy Loss

'''Instantiate the neural network, set our learning rate, and instantiate optimizer.'''

model=FullyConnectedNetwork().to(device);
#loss_criterion = F.nll_loss;
loss_criterion = nn.CrossEntropyLoss()
learning_rate = 0.001;
# note that we have to add all weights&biases, for both layers, to the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
n_epochs = 15;
num_updates = n_epochs*int(np.ceil(len(trainloader.dataset)/batch_size))
# warmup_steps=1000;
#def warmup_linear(x):
#    if x < warmup_steps:
#        lr=x/warmup_steps
#    else:
#        lr=max( (num_updates - x ) / (num_updates - warmup_steps), 0.)
#    return lr;
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear);

'''Training Loop'''
print('Entering Training Loop:')
for i in range(n_epochs):
    for j, (inputs, labels) in enumerate(trainloader):
      
        inputs=inputs.to(device);
        labels = labels.type(torch.LongTensor)
        labels=labels.to(device);
        

        #forward phase - predictions by the model
        outputs = model(inputs);
        loss = loss_criterion(outputs, labels)

        # calculate gradients
        optimizer.zero_grad();
        loss.backward();

        # take the gradient step
        optimizer.step();
#        scheduler.step();

        batch_loss=loss.item();  
    with (torch.no_grad()):
        
        correct = 0;
        for j, (inputs, labels) in enumerate(trainloader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)
            #print(outputs)
            pred = outputs.data.max(dim=1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
        training_accuracy = correct/len(trainloader.dataset)
        correct = 0;
        for j, (inputs, labels) in enumerate(testloader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)
            #print(outputs)
            pred = outputs.data.max(dim=1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
        testing_accuracy = correct/len(testloader.dataset)
    print('Epoch: '+str(i+1)+'/'+str(n_epochs)+' | Loss: '+str(batch_loss)+
          ' | Training Accuracy: '+str(training_accuracy)+
          ' | Testing Accuracy: '+str(testing_accuracy))

