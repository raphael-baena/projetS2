import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable


# Hyper Parameters
K=4
N=7
N_time=10
num_epochs = 3000
test_size=1000
batch_size = 10000
real_batch_size = 3000
learning_rate = 0.01
FC=400

#DATA TRAIN
loaded_npz = np.load("data_input.npz")
x_train,r_train = loaded_npz["data"]
print(x_train.shape)
x_train=x_train.reshape(batch_size,N_time,K,N)
r_train=r_train.reshape(batch_size,N_time,K,N)
loaded_npz = np.load("data_label.npz")
l_train=loaded_npz["data"]
l_train.reshape(batch_size,N_time,K,N)
#
#batch_size = 1000
#l_train = l_train[:batch_size,:]
#x_train = x_train[:batch_size,:,:,:]

tensor_x_train = torch.stack([torch.Tensor((i-np.mean(i))/np.std(i))for i in x_train]) # transform to torch tensors
tensor_l_train = torch.stack([torch.Tensor(i) for i in l_train])
tensor_r_train = torch.stack([torch.Tensor(i) for i in r_train])

train_dataset = torch.utils.data.TensorDataset(tensor_x_train,tensor_l_train)
###DATA TEST
loaded_npz = np.load("data_input_test.npz")
x_test,r_test = loaded_npz["data"]
x_test=x_test.reshape(test_size,N_time,K,N)
r_test=r_test.reshape(test_size,N_time,K,N)
loaded_npz = np.load("data_label_test.npz")
l_test=loaded_npz["data"]
l_test.reshape(test_size,N_time,K,N)
tensor_x_test = torch.stack([torch.Tensor((i-np.mean(i))/np.std(i))for i in x_test]) # transform to torch tensors
tensor_l_test = torch.stack([torch.Tensor(i) for i in l_test])
tensor_r_test= torch.stack([torch.Tensor(i) for i in r_test])
test_dataset = torch.utils.data.TensorDataset(tensor_x_test,tensor_l_test)
####

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(batch_size=real_batch_size, dataset=train_dataset)

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
        self.fc2=nn.Linear(FC,FC)
        self.fc3=nn.Linear(FC,FC)
        self.fc4=nn.Linear(FC,FC)
        self.fc5 = nn.Linear(FC,N_time*K*N)
        self.do = nn.Dropout()
        
    def forward(self, x):
#        out = self.layer1(x)
#        out = self.layer2(out)
        out = x.view(-1,N_time*K*N)
        out = self.do(nn.functional.relu(self.fc1(out)))
        
        out =self.do(nn.functional.relu(self.fc2(out)))
        out =self.do(nn.functional.relu(self.fc3(out)))
        out =self.do(nn.functional.relu(self.fc4(out)))
        out = self.fc5(out)
        out=out.view(-1,N_time,K,N)
        out=nn.Softmax(2)(out)
        out =out.view(-1,N_time*K*N)
        return out
        
cnn = CNN()
cnn=cnn.cuda()
#
## Loss and Optimizer
#criterion =  nn.KLDivLoss()
criterion=nn.MSELoss()
criterion_test=nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#
## Train the Model
#
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

def winner_take_it_all(outputs):
    
    for t in range(outputs.size(1)):
            _,m=torch.max(outputs[0,t,:,:],dim=0)
            for s in range(outputs.size(3)):
                for k in range(outputs.size(2)):
                    if k==int(m[s]):
                        outputs[0,t,k,s]=1
                    else:
                        outputs[0,t,k,s]=0
    return 
                

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #data_rata=tensor_r[i,:,:]
        optimizer.zero_grad()

        images = Variable(images,requires_grad=True).cuda()
        labels = Variable(labels).cuda()
#        labels=labels.view(-1,N_time,K,N)
#        # Forward + Backward + Optimize
        outputs = cnn(images)
        
        
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        if i+1 == batch_size // real_batch_size:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//real_batch_size, loss.data[0]))
            ## Test the Model
            cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
            correct = 0
            total = 0
            loss=0
            for i, (images, labels) in enumerate(test_loader):
                images = Variable(images,requires_grad=True).cuda()
                labels = Variable(labels).cuda()
                labels=labels.view(-1,N_time,K,N)
                outputs = cnn(images)
                loss = criterion_test(outputs,labels)
                if (i+1) % 1000 == 0:
                    print (len(train_dataset)//1000, loss.data[0])        


#print(outputs)
## Test the Model
cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
loss=0
for i, (images, labels) in enumerate(test_loader):
    images = Variable(images,requires_grad=True).cuda()
    labels = Variable(labels).cuda()
    labels=labels.view(-1,N_time,K,N)
    outputs = cnn(images)
    loss += criterion_test(outputs,labels)
    if (i+1) % 1000 == 0:
        print (len(train_dataset)//1000, loss.data[0]/(i+1))
outputs=outputs.view(-1,N_time,K,N)
print(outputs)

images = Variable(tensor_x_train[0,:,:,:],requires_grad=True).cuda()
labels = Variable(tensor_l_train[0,:]).cuda()
labels=labels.view(-1,N_time,K,N)
outputs = cnn(images)
outputs=outputs.view(-1,N_time,K,N)
print(outputs)
print(labels)

