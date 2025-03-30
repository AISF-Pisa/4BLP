import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torchneural import NN
from neural import NeuralNetworkRegressor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(69420)
np.random.seed(69420)

class PINN:
    '''
    Physics Informed Neural Network for solving an ODE
    '''

    def __init__(self, layers, t_min, t_max, gamma, omg_0, act=nn.Tanh()):
        '''
        Parameters
        ----------
        layers : list
            List with the number of neurons for each hidden layer
        t_min : float
            Minimum value of time
        t_max : float
            Maximum value of time
        m : float
            Mass of the oscillator
        c : float
            Damping coefficient
        k : float
            Spring constant
        act : torch function, optional
            Activation function of the network
        '''
        self.net = NN(dim_in=1, dim_out=1, layers=layers, r_max=t_max, r_min=t_min, act=act).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.gamma = gamma
        self.omg_0 = omg_0

    def f(self, t):
        ''' ODE we want to solve in the form f(x, t) = 0 
        '''
        t = t.clone()
        t.requires_grad = True

        x    = self.net(t)                                              # predicted solution x(t)
        x_t  = torch.autograd.grad(x.sum(), t, create_graph=True)[0]    # dx/dt
        x_tt = torch.autograd.grad(x_t.sum(), t, create_graph=True)[0]  # d^2x/dt^2

        ODE = x_tt + self.gamma* x_t + self.omg_0 * x

        return ODE

    def train(self, n_epoch, t_0, y_0, v_0, domain_f, data_train, data_target):
        '''
        Train the network

        Parameters
        ----------
        n_epoch : int
            Number of epochs for training
        t_0 : torch.tensor
            Initial time
        y_0 : torch.tensor
            Initial position
        v_0 : torch.tensor
            Initial speed
        domain_f : torch.tensor
            Collocation points for evaluating the ODE
        data_train : torch.tensor
            data for training
        data_target : torch.tensor
            target for training data
        
        Returns
        -------
        Loss : list
            List of training loss over epochs
        '''
        Loss = []
        for epoch in range(n_epoch):

            self.optimizer.zero_grad()  # zero gradients

            # Loss form data
            u_pred = self.net(data_train)
            mse_data = torch.mean(torch.square(u_pred - data_target))

            # Loss from initial condition on position
            y0_pred = self.net(t_0)
            y0_mse  = torch.mean(torch.square(y0_pred - y_0))

            # Loss from initial condition on speed
            v_t0_pred = torch.autograd.grad(y0_pred.sum(), t_0, create_graph=True)[0]
            v0_mse    = torch.mean(torch.square(v_t0_pred - v_0))

            # Loss from ODE
            f_pred = self.f(domain_f)
            ode_mse = torch.mean(torch.square(f_pred))

            loss = mse_data + y0_mse + v0_mse + ode_mse
            
            loss.backward()
            self.optimizer.step()

            with torch.autograd.no_grad():
                Loss.append(loss.data.detach().cpu().numpy())
                if epoch % 100 == 0:
                    print(f"epoch: {epoch}, data: {mse_data:.3e}, x_0: {y0_mse:.3e}, v_0: {v0_mse:.3e}, ODE: {ode_mse:.3e}")
        
        return Loss

#=======================================================
# Computational parameters for ODE
#=======================================================

# Time interval
t_min      = 0.0
t_max      = 10.0
t_data_min = 0.0
t_data_max = 4.0

# Number of points for collocation
N_t = 1000
# Number of points in dataset
N_p = 100

# Parameters for the system
gamma = 0.5                   # damping coefficient
omg   = np.sqrt(2*np.pi)      # final omega
omg_0 = omg**2 + (gamma/2)**2 # omega of the equation

# Initial conditions
t_0 = np.array([[0.0]]) # Initial time t = 0
y_0 = np.array([[1.0]]) # x(0) = 1
v_0 = np.array([[0.0]]) # v(0) = 0

# Collocation points in the time domain
t_f = np.random.uniform(t_min, t_max, (N_t, 1))

# Creation of the dataset
X = np.linspace(t_data_min, t_data_max, N_p)
n = len(X)
y = np.cos(omg*X)*np.exp(-X*gamma/2)   # target

X = X[:, None]
y = y[:, None]

# split dataset in test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42)

#=======================================================
# Convert to Tensor
#=======================================================

# requires_grad = True will be necessary for initial speed loss
# in general it is necessary when we want to compute derivatives
t_0        = torch.tensor(t_0, dtype=torch.float, requires_grad=True).to(device)
y_0        = torch.tensor(y_0, dtype=torch.float).to(device)
v_0        = torch.tensor(v_0, dtype=torch.float).to(device)
domain_f   = torch.tensor(t_f, dtype=torch.float).to(device)
X_train_t  = torch.tensor(X_train, dtype=torch.float).to(device)
Y_train_t  = torch.tensor(Y_train, dtype=torch.float).to(device)

#=======================================================
# Creation of network and train
#=======================================================

n_epoch = 12000 + 1
pinn = PINN([50, 50, 50], t_min, t_max, gamma, omg_0)
Loss = pinn.train(n_epoch, t_0, y_0, v_0, domain_f, X_train_t, Y_train_t)

#=======================================================
# Plot the loss function
#=======================================================
plt.figure()
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid()
plt.plot(range(n_epoch), Loss)

#=======================================================
# Prediction of PINN
#=======================================================

t = np.linspace(t_min, t_max, 1000)
t_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float).to(device)
x_analytical = np.exp(-gamma/2 * t) * np.cos(omg * t)

# Disable gradient calculation for plotting
with torch.no_grad():
    y_pred = pinn.net(t_tensor).cpu().numpy()

#=============================================================
# Creation of dataset for simple NN
#=============================================================

X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train[:, 0], Y_test[:, 0] 

# for plot
Xp = np.linspace(t_min, t_max, 1000)
Xp = np.expand_dims(Xp, axis=1)

#=============================================================
# Creation of models
#=============================================================
n_epoch = 10000 + 1
lr_rate = 0.01

NN = NeuralNetworkRegressor([50, 50, 50], n_epoch, f_act='tanh')

result = NN.train(X_train, Y_train, alpha=lr_rate, verbose=True)

L_t = result['train_Loss']
L_v = result['valid_Loss']

#=============================================================
# Plot Loss
#=============================================================

plt.figure(3)
plt.plot(np.linspace(1, n_epoch, n_epoch), L_t, 'b', label='train Loss')
plt.plot(np.linspace(1, n_epoch, n_epoch), L_v, 'r', label='validation loss')
plt.title('Binay cross entropy', fontsize=15)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(loc='best')
plt.grid()

#=============================================================
# Plot
#=============================================================

p_fit = NN.predict(Xp.T)
p_fit = p_fit[0]

plt.figure(2, figsize=(9, 5))
plt.title('Network regression result', fontsize=15)
plt.xlabel('t', fontsize=15)
plt.ylabel('y(t)', fontsize=15)
plt.grid()
plt.errorbar(X_train.T[:,0], Y_train, fmt='.', c='r', label='train data')
plt.errorbar(X_test.T[:,0],  Y_test, fmt='.', c='b', label='test data')
plt.plot(Xp[:,0], p_fit,'k', label='NN')
plt.plot(t, y_pred, label='PINN', color='darkviolet')
plt.plot(t, x_analytical, label='Analytical', linestyle='--', color='green')
plt.legend(loc='best')
plt.show()
