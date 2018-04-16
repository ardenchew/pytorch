import torch
import torch.nn as nn
import torchvision.datasets
from torch.autograd import Variable

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

##TO-DO: Import data here:
loader_train = DataLoader(MNIST(root='datasets/',train=False,download=False,
                                transform=transforms.ToTensor()), batch_size=100, shuffle=True)


##TO-DO: Define your model:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        num_features = 784
        num_l1 = 300
        num_l2 = 150
        num_c = 10
        
        self.layer_1 = nn.Linear(num_features,num_l1)
        self.layer_2 = nn.Linear(num_l1,num_l2)
        self.layer_out = nn.Linear(num_l2,num_c)

        ##Define layers making use of torch.nn functions:
    
    def forward(self, x):

        ##Define how forward pass / inference is done:
        m = nn.ReLU()
        output = self.layer_1(x)
        output = m(output)
        output = self.layer_2(output)
        output = m(output)
        output = self.layer_out(output)
        return output

my_net = Net()
loss = nn.CrossEntropyLoss()
sgd = torch.optim.SGD(my_net.parameters(),lr=0.01)


##TO-DO: Train your model:
for iter in range(50):
    my_net.train()
    for idx, (features, label) in enumerate(loader_train):
        sgd.zero_grad()
        features = Variable(features.view(-1,28*28))
        label = Variable(label)
        pred = my_net(features)
        curr_loss = loss(pred,label)
        curr_loss.backward()
        sgd.step()
        
torch.save(my_net.state_dict(),'model.pkl')
