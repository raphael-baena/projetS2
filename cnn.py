import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable


# Hyper Parameters
K=10#number of user
N=20#number of subcarrier
N_time=12#number of time slot
num_epochs = 100
batch_size = 10000
learning_rate = 0.00001
test_size=1000
FC=1000

### TRAIN Data set
loaded_npz = np.load("data_input.npz")
x,r = loaded_npz["data"]
x=x.reshape(batch_size,N_time,K,N)
r=r.reshape(batch_size,N_time,K,N)
loaded_npz = np.load("data_label.npz")
l=loaded_npz["data"]
l.reshape(batch_size,N_time,K,N)
tensor_x = torch.stack([torch.Tensor((i-np.mean(i))/np.std(i))for i in x]) # Data Standarization 
tensor_l = torch.stack([torch.Tensor(i) for i in l])#label tensor
tensor_r= torch.stack([torch.Tensor(i) for i in r])#

train_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_l)

### TEST Data set
loaded_npz = np.load("data_input_test.npz")
x,r = loaded_npz["data"]
x=x.reshape(test_size,N_time,K,N)
r=r.reshape(test_size,N_time,K,N)
loaded_npz = np.load("data_label_test.npz")
l=loaded_npz["data"]
l.reshape(test_size,N_time,K,N)
tensor_x = torch.stack([torch.Tensor((i-np.mean(i))/np.std(i))for i in x]) # transform to torch tensors
tensor_l = torch.stack([torch.Tensor(i) for i in l])
tensor_r= torch.stack([torch.Tensor(i) for i in r])

test_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_l)
####

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
#

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
#        self.layer1 = nn.Sequential(
#            nn.Conv2d(12, 12, kernel_size=5, padding=2),
#            nn.BatchNorm2d(12),
#            nn.functional.relu),
#            nn.MaxPool2d(1))
#        self.layer2 = nn.Sequential(
#            nn.Conv2d(12, 12, kernel_size=5, padding=2),
#            nn.BatchNorm2d(12),
#            nn.functional.relu),
#            nn.MaxPool2d(1))
        self.fc1=nn.Linear(N_time*K*N,FC)
        self.fc2=nn.Linear(FC,2*FC)
        self.fc3=nn.Linear(2*FC,3*FC)
        self.fc4=nn.Linear(3*FC,4*FC)
        self.fc5 = nn.Linear(2*FC,N_time*K*N)
        
    def forward(self, x):
#        out = self.layer1(x)
#        out = self.layer2(out)
        out = x.view(-1,N_time*K*N)
        out =nn.functional.relu(self.fc1(out))

        out =nn.functional.relu(self.fc2(out))
       # out =nn.functional.relu(self.fc3(out))
        #out =nn.functional.relu(self.fc4(out))
        out = nn.functional.relu(self.fc5(out))
        out=out.view(-1,N_time,K,N)
        out=nn.Softmax(2)(out)
        out=nn.Threshold(1/K,0)(out)
        return out
        
cnn = CNN()
cnn=cnn.cuda()
#
## Loss and Optimizer
criterion = nn.BCELoss()#criterion that measures the Binary Cross Entropy between the target and the output:
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
               

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        images = Variable(images,requires_grad=True).cuda()
        labels = Variable(labels).cuda()
        labels=labels.view(-1,N_time,K,N)
        outputs = cnn(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1000 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
# Test the Model
cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
loss=0
for i, (images, labels) in enumerate(test_loader):
    images = Variable(images,requires_grad=True).cuda()
    labels = Variable(labels).cuda()
    labels=labels.view(-1,N_time,K,N)
    outputs = cnn(images)
    loss += criterion(outputs,labels)
    if (i+1) % 1000 == 0:
        print (len(train_dataset)//1000, loss.data[0]/(i+1))

#USELESS CODE AT THE MOMENT
#def Tk_Network_Output(R,A):
#    N_time=np.size(R,0)  #nb timeslot
#    A=A.data.numpy()
#    K=np.size(R,1) 
#    Tk=np.zeros((N_time,K))
#    tc=20
#    for t in range(N_time):
#        A_t=np.transpose(A[t,:,:])
#        Rtot=np.diagonal(np.matmul(R[t,:,:],A_t))
#        if t>1:
#            Tk[t,:]=Tk[t-1,:]*(1-1/tc)
#        Tk[t,:]=Tk[t,:]+Rtot/tc
#    return Tk
#
#def winner_take_it_all(outputs):
#    
#    for t in range(outputs.size(1)):
#            _,m=torch.max(outputs[0,t,:,:],dim=0)
#            for s in range(outputs.size(3)):
#                for k in range(outputs.size(2)):
#                    if k==int(m[s]):
#                        outputs[0,t,k,s]=1
#                    else:
#                        outputs[0,t,k,s]=0
#    return 
