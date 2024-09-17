import numpy as np
import sklearn as sk

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
# Hyperparameters
# Feel free to tune these hyperparameters, 
# as long as they are kept optimal/fair for both base and improved model
batch_size = 256
epochs = 10

######################################################################
# Create the Dataset and DataLoader (both for training and testing)
######################################################################
# Hint: three things are typically done to the data 
#   1. create the data transform object for the training/testing data
#   2. create the dataset object
#   3. create the dataloader object
######################################################################
# Here is a simple code structure for the CIFAR10 training dataset, please complete the following code.
# train_transform = transforms.Compose([???])
# train_dataset = torchvision.datasets.CIFAR10(???)
# train_loader = torch.utils.data.DataLoader(???)
#######################################################################

# for the training and test transform, here we only need two transforms:
#   1. convert the image to tensor
#   2. normalize the tensor with mean=(0.5, 0.5, 0.5) and variance=(0.5, 0.5, 0.5)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# create the training and test dataset with torchvision.datasets.CIFAR10() class.
train_dataset = datasets.CIFAR10(root=".", download=False, train = True, transform= transform)
test_dataset = datasets.CIFAR10(root=".", download=False, train = False, transform= transform)

# create the training and test dataloader with torch.utils.data.DataLoader() class.
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, )
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# the 10 classes in the CIFAR10 dataset.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Base Model Definition
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    """
    Structure of the model: 
        1. Conv2D layer with (in_channels=3, out_channels=6, kernel_size=5)
        2. ReLU activation
        3. Max Pooling layer with (kernel_size=2, stride=2)
        4. Conv2D layer with (in_channels=6, out_channels=16, kernel_size=5)
        5. ReLU activation
        6. Max Pooling layer with (kernel_size=2, stride=2)
        7. Flatten layer
        8. Fully connected layer with (in_features=???, out_features=120)
        9. ReLU activation
        10. Fully connected layer with (in_features=???, out_features=84)
        11. ReLU activation
        12. Fully connected layer with (in_features=???, out_features=???) # the output size is the number of classes
    """
    def __init__(self):
        super().__init__()
        self.Conv2D1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2D2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        

    def forward(self, x):
        x = self.Conv2D1(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.Conv2D2(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = F.relu(x)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

# create the model and move it to GPU for training acceleration
# base_model = ???
base_model = BaseModel()

# create the criterion, which should be ??? for multi-class classification?
# criterion = ???
criterion = nn.CrossEntropyLoss()

import torch.optim as optim
# create the optimizer:
# SGD optimizer with learning rate 0.001 and momentum 0.9
# optimizer = ???
optimiser = optim.SGD(base_model.parameters(), lr=0.001, momentum=0.9) 

# Network Training
from tqdm import tqdm

for epoch in range(epochs):  # loop over the dataset multiple times
    losses = []
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0], data[1]

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward the output
        y_hat = base_model(inputs)

        # compute the loss with the criterion object
        running_loss = criterion(y_hat, labels)

        # backward the loss
        running_loss.backward()

        # update the model
        optimiser.step()

        # TODO: log the training losses, accuracies
        losses.append(running_loss)
        if i%200 == 0:
            print('epoch:',epoch,'\t','instance:',i,'\t','Avg loss:',sum(losses[-200:])/200)
            
        # TODO: log the validation losess, accuracies
    with torch.no_grad():
        accs = []
        for i, data in enumerate(tqdm(testloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0], data[1]

            # forward the output
            y_hat = base_model(inputs)
            # TODO: log the training losses, accuracies
            acc = (np.argmax(y_hat.detach().numpy(), axis = 1) == labels.numpy()).astype(int).mean()
            accs.append(acc)

    print('epoch:',epoch,'\t','instance:',i,'\t','Avg acc:',sum(accs)/len(accs))
                        

# after training, let's evaluate the model on the test dataset, 
# with our previously implemented "clf_report" function.
# print('Base Model on Test Data:\n', clf_report(???, ???))
pass