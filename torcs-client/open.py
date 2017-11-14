import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class NET(nn.Module):

    def __init__(self, in_size, out_size, hidden_size,  parameters_file=None):
        super(NET, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(in_size, hidden_size, bias = False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias = False)
        self.h2o = nn.Linear(hidden_size + in_size, out_size, bias = False)
        self.sig = nn.Sigmoid()

        sp_rad = max(abs(np.linalg.eig(self.h2h.weight.data.numpy())[0]))
        self.h2h.weight.data *= 1/sp_rad

        #if parameters_file is not None:

    def forward(self, input, hidden):
        a1 = self.i2h(input)
        a2 = self.h2h(hidden)
        hidden = self.sig(a1 + a2)
        output = self.h2o(torch.cat((hidden.squeeze(), input.squeeze())))

        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    #def save_parameters(self, file):
    #    np.savetxt(file, self.parameters())

torch.manual_seed(42)

f = open('./train_data/alpine-1.csv', 'r')
lines = f.readlines()

input_size = len(lines[0].split(",")) - 3
output_size = 3
N = 8000
N_max = len(lines)-2
hidden_size = 1000  # size of reservoir
washout = 100

input_train = np.zeros((N, input_size))
target_train = np.zeros((N, output_size))

input_test = np.zeros((N_max - N, input_size))
target_test = np.zeros((N_max - N, output_size))

for ind, line in enumerate(lines[1:N_max]):
    if ind < N:
        target_train[ind] = line.split(",")[:3]
        input_train[ind] = line.split(",")[3:-1]
    else:
        target_test[N-ind] = line.split(",")[:3]
        input_test[N-ind] = line.split(",")[3:-1]

model = NET(input_size, output_size, hidden_size)

#collect states
X = np.zeros((N-washout, input_size+hidden_size))
Yt = target_train[washout:N]

# run the reservoir with the data and collect X
hidden = model.init_hidden()
for t in range(N):
    net_input = Variable(torch.FloatTensor(input_train[t]))
    output, hidden = model.forward(net_input, hidden)
    hidden_nump = hidden.data.numpy()
    input_nump = net_input.data.numpy()
    if t >= washout:
        a = np.concatenate([hidden_nump.flatten(), input_nump.flatten()])
        X[t - washout, :] = a

# train
X_T = X.T
pseud_S = np.linalg.inv(X_T @ X) @ X_T
transposed = pseud_S @ Yt
model.h2o.weigth = transposed.T

loss = nn.MSELoss()

#print(target_test)
for i in range(N_max - N):
    test_input = Variable(torch.FloatTensor(input_test[i]))
    output, hidden = model.forward(test_input, hidden)
    target = Variable(torch.FloatTensor(target_test[i]))
    err = loss(output, target)
    print(i, output, err)
