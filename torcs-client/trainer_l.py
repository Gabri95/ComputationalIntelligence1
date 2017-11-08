import random
import torch
from torch.autograd import Variable
import torch.nn as nn


dtype = torch.FloatTensor



class RNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, init_res_state = None):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        if init_res_state is not None:
            self.hidden_layer = init_res_state
        else:
            self.hidden_layer = Variable(torch.rand(self.hidden_size), requires_grad = False)

        self.w_in = Variable(torch.rand(self.input_size, self.hidden_size), requires_grad = False)
        self.w_hid = Variable(torch.rand(self.hidden_size, self.hidden_size), requires_grad = False)
        self.w_out = Variable(torch.rand(self.hidden_size, self.hidden_size), requires_grad = True)
        self.w_back = Variable(torch.rand(self.hidden_size, self.output_size), requires_grad=False)


    def forward(self, inputs):

        x_n_1 = self.hidden_layer.clone()

        u_n = self.w_in @ inputs
        x_n = self.w_hid @ x_n_1
        x_out = nn.Sigmoid(u_n + x_n)
        y_out = nn.Sigmoid(self.w_out @ x_out)

        return y_out

#test

input_size = 20
output_size = 3
N = 10

file = open('./train_data/aalborg.csv')


in_list=[]
out_list=[]

for line in file.readlines():
    ls = [val for val in line.split(",")]
    in_list.append(ls[:3])
    out_list.append(ls[3:])

input = Variable(torch.Tensor(np.array(in_list).reshape(N,input_size)))
output = Variable(torch.Tensor(np.array(out_list).reshape(N,out_size)))























