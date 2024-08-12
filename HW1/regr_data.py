#####################################################
##  ME8813ML Homework 1: 
##  Implement a quasi-Newton optimization method for data fitting
## 
## Name: Donaven Lobo
#####################################################
import numpy as np
import matplotlib.pyplot as plt


########################################################
## Implement a parameter fitting function fit() so that 
##  p = DFP_fit(x,y)
## returns a list of the parameters as p of model:
##  p0 + p1*cos(2*pi*x) + p2*cos(4*pi*x) + p3*cos(6*pi*x)  
########################################################

# GLOBAL VARIABLES:
PI = np.pi


# Fixing random state for reproducibility
np.random.seed(19680801)

dx = 0.1
x_lower_limit = 0
x_upper_limit = 40                                       
x = np.arange(x_lower_limit, x_upper_limit, dx)
data_size = len(x)                                 # data size
noise = np.random.randn(data_size)                 # white noise

# Original dataset 
y = 2.0 + 3.0*np.cos(2*np.pi*x) + 1.0*np.cos(6*np.pi*x) + noise


###########################################
initial_p = [1, 1, 1, 1]  # Initial parameter guess
eps = 1e-6  # Convergence threshold
# p = DFP_fit(x, y, initial_p, eps)
###########################################


fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y)
axs[0].set_xlim(x_lower_limit, x_upper_limit)
axs[0].set_xlabel('x')
axs[0].set_ylabel('observation')
axs[0].grid(True)


#########################################
## Plot the predictions from your fitted model here
axs[1].set_xlim(x_lower_limit, x_upper_limit)
axs[1].set_xlabel('x')
axs[1].set_ylabel('model prediction')

fig.tight_layout()
plt.show()

def DPF_fit(x,y,initial_p,eps):
#     Implement this function to calculate the p values

    
    base = np.array([np.ones_like(x), np.cos(2*PI*x), np.cos(4*PI*x), np.cos(6*PI*x)]) # Base values of the objective funtion
    grad = np.array([np.zeros_like(x), -2*PI*np.sin(2*PI*x), -4*PI*np.sin(4*PI*x), -6*PI*np.sin(6*PI*x)]) # Gradient of objective function
    
    y_p = np.multiply(base,p)

    # Set the initial Heissen Matrix inverse approx (B_0) as the identity matrix
    B_0 = np.identity(len(p))
    
    while np.linalg.norm(grad) > eps:
        
    
    
def stop_crit_check(x):
#     Create a function that calculates the value to compare with epsilon

def loss_func(p):
#     Return the error calculated by the loss function
    

    

    
