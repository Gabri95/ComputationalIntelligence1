
import numpy as np

class Model():
    def __init__(self, weights_file):
        file = open(weights_file, 'r')
        lines = file.readline()

        # the first line contains the number of input neurons and output neurons, separated by a comma
        self.I, self.O = (int(n) for n in lines[0].split(','))
        
        # Matrix containing the connections from each neuron (input, hidden or output) to each hidden or output neuron.
        # As a result, the size has to be (H+O)*(I+H+O).
        # Moreover, the neurons have to be in the following order: input, output, hidden.
        self.W = np.genfromtxt(weights_file, skip_header=1)

        # from the shape of the parameters we can infer the number of hidden neurons
        self.H = self.W.shape[0] - self.O
        
        assert self.H == self.W.shape[1] - self.I - self.O, "Error! Shape of the parameters not valid!"

        # array containing the activiation values of each of the neurons
        # N.B. input neurons are excluded
        # N.B the first O neurons are the output ones, while the last H ones are the hidden ones
        self.V = np.zeros(self.O + self.H)

    def __init__(self, I, O, H):
        
        #set the shape of the network
        self.I, self.O, self.H = I, O, H
    
    
        # Matrix containing the connections from each neuron (input, hidden or output) to each hidden or output neuron.
        # As a result, the size has to be (H+O)*(I+H+O).
        # Moreover, the neurons have to be in the following order: input, output, hidden.
        
        #initialize randomly the parameters
        self.W = np.random.normal(0, 0.1, (H+O, I+H+O))

        

        # array containing the activiation values of each of the neurons
        # N.B. input neurons are excluded
        # N.B the first O neurons are the output ones, while the last H ones are the hidden ones
        self.V = np.zeros(self.O + self.H)
        
    
    #return the total number of nodes in the network, including also the input layer
    def networkSize(self):
        return self.I + self.O + self.H
    
    #return the number of actual neurons in the network (i.e. hidden + output neurons)
    def numberOfNeurons(self):
        return self.O + self.H
        
    #propagate the input for one step in the network and returns the new values in the output layer
    #N.B.: no activation function is applied to the output layer, i.e. the output layer has a linear (identity) activation function
    def step(self, input):
        #propagate the value of every neuron to the ones it is connected to
        self.V = self.W.dot(np.concatenate([input, self.V]))
        
        #apply activation function
        # to the hidden neurons
        self.V[self.O:] = np.tanh(self.V[self.O:])

        # to the output neurons
        #self.V[:self.O] = np.tanh(self.V[:self.O])
        
        #return the output layer
        return self.V[:self.O]
    
    #save the parameters of the network to the specified file, in the correct format (the one accepted in the constructor)
    def save_to_file(self, file):
        np.savetxt(file, self.W, header=str(self.I) + ', ' + str(self.O))
    
    
#Sigmoid Function
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))