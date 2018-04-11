import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable


# Hyper Parameters
K=5
N=7
N_time=12
num_epochs = 5
batch_size = 10
learning_rate = 0.001
FC=1000

#Data
loaded_npz = np.load("data_input.npz")
x,r = loaded_npz["data"]
x=x.reshape(batch_size,N_time,K,N)
r=r.reshape(batch_size,N_time,K,N)
loaded_npz = np.load("data_label.npz")
l=loaded_npz["data"]
l.reshape(batch_size,N_time,K)


tensor_x = torch.stack([torch.Tensor(i) for i in x]) # transform to torch tensors
tensor_l = torch.stack([torch.Tensor(i) for i in l])
tensor_r= torch.stack([torch.Tensor(i) for i in r])

train_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_l)
test_dataset = torch.utils.data.TensorDataset(torch.Tensor(x),torch.Tensor(l))


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                     shuffle=False)
#
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
#        self.layer1 = nn.Sequential(
#            nn.Conv2d(12, 12, kernel_size=5, padding=2),
#            nn.BatchNorm2d(12),
#            nn.ReLU(),
#            nn.MaxPool2d(1))
#        self.layer2 = nn.Sequential(
#            nn.Conv2d(12, 12, kernel_size=5, padding=2),
#            nn.BatchNorm2d(12),
#            nn.ReLU(),
#            nn.MaxPool2d(1))
        self.fc1=nn.Linear(N_time*K*N,FC)
        self.fc2=nn.Linear(FC,2*FC)
        self.fc3=nn.Linear(2*FC,3*FC)
        self.fc4=nn.Linear(3*FC,4*FC)
        self.fc5 = nn.Linear(4*FC,N_time*K*N)
        
    def forward(self, x):
        #out = self.layer1(x)
        #out = self.layer2(out)
        out = x.view(N_time*K*N)
        out =self.fc1(out)
        out =self.fc2(out)
        out =self.fc3(out)
        out =self.fc4(out)
        out = self.fc5(out)
        out=x.view(N_time,K,N)
        out=nn.Softmax(1)(out)
        return out
        
cnn = CNN()
cnn
#
## Loss and Optimizer
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#
## Train the Model

def Tk_Network_Output(R,A):
    N_time=np.size(R,0)  #nb timeslot
    K=np.size(R,1) 
    Tk=np.zeros((N_time,K))
    tc=20
    for t in range(N_time):
        A_t=np.transpose(A[t,:,:])
        Rtot=np.diagonal(np.matmul(R[t,:,:],A_t))
        if t>1:
            Tk[t,:]=Tk[t-1,:]*(1-1/tc)
        Tk[t,:]=Tk[t,:]+Rtot/tc
    return Tk

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        data_rata=tensor_r[i,:,:]
        images = Variable(images)
        labels = Variable(labels)
#        
#        # Forward + Backward + Optimize
#        optimizer.zero_grad()
        outputs = cnn(images)
        
        """The out put must be convert into Tk to be compare with labels"""
#        
#        
#        
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#        
#        if (i+1) % 100 == 0:
#            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
#                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

## Test the Model
#cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
#correct = 0
#total = 0
#for images, labels in test_loader:
#    images = Variable(images).cuda()
#    outputs = cnn(images)
#    _, predicted = torch.max(outputs.data, 1)
#    total += labels.size(0)
#    correct += (predicted.cpu() == labels).sum()
#
#print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
#