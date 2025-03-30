"""
Code to create the neural network architecture
that will then be used in codes to solve differential equations.
"""
import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(69420)


class Layer(nn.Module):
    ''' Class for NN layers
    '''
    def __init__(self, n_in, n_out, act):
        '''
        Creation of the layers

        Parameters
        ----------
        n_in : int
            number of neurons of the previous layer
        n_out : int
            number og neurons of the current layer
        act : torch.nn function
            activation functionn of the layer
        '''
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.act   = act

    def forward(self, x):
        '''
        Feedforward through the single layer

        Parameter
        ---------
        x : torch.tensor
            input of the layer
        '''
        x = self.layer(x)
        
        if self.act:
            x = self.act(x)
        return x


class NN(nn.Module):
    """
    Class for the neural network
    """
    def __init__(self, dim_in, dim_out, layers, r_max, r_min, act):
        '''

        Parameters
        ----------
        dim_in : int
            dimension of the input
        dim_out : int
            dimension of the output
        layers : list
            list which must contain the number of neurons for each layer
            the number of layers is len(layers) and layers[i] is the 
            number of neurons on the i-th layer. Only hidden layers must
            be declared
        r_max : torch.tensor
            max value of the input parameters
            e.g. if we are in the square  0<x<1 0<y<1 r_max = [1, 1] 
        r_min : torch.tensor
            min value of the input parameters
            e.g. if we are in the square  0<x<1 0<y<1 r_max = [0, 0]
        act : torch.nn function
            activation functionn of the layer
        '''
        super().__init__()
        self.net = nn.ModuleList()

        layers = layers + [layers[-1]]   # To obtain the exact number of hidden layes 

        self.net.append(Layer(dim_in, layers[0], act))            # Input layer
        for i in range(1, len(layers)):                           # Hidden layer
            self.net.append(Layer(layers[i-1], layers[i], act))
        self.net.append(Layer(layers[-1], dim_out, act=None))     # Output layer

        self.r_max = torch.tensor(r_max, dtype=torch.float).to(device)
        self.r_min = torch.tensor(r_min, dtype=torch.float).to(device)
        
        self.net.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.Linear):
            torch.nn.init.xavier_uniform_(l.weight.data)
            torch.nn.init.zeros_(l.bias.data)

    def forward(self, x):
        '''
        Feedforward through the all network

        Parameter
        ---------
        x : torch.tensor
            input of the layer
        '''

        out = (x - self.r_min) / (self.r_max - self.r_min)  # Min-max scaling
        
        for layer in self.net: # loop over all layers
            out = layer(out)
        return out