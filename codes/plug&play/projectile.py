# Import libraries
import numpy as np
import time
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, hessian, jacfwd
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import get_params_format_fn
from flax.struct import dataclass
from typing import Tuple
from flax import linen as nn
from new_sim_mgr import SimManager
from scipy import sqrt, diag, cos, sin, pi
# choose GPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Functions and data
# Contains the train input and its labels
@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray

# Function to generate analytical solution
def f_x(t):
    return x0 + t*u0

def f_y(t):
    return y0 + t*v0 - 1./2*g*t**2

def f_t():
    return (vel0*sin(a0*pi/180) + sqrt((vel0*sin(a0*pi/180))**2 + 2*g*y0)) / g

# Specify planet to change g, d & a_T
planet = {'earth': (9.8, 0, 2), 'mars': (3.7, 0, 5.5), 'moon': (1.6, 0, 8)}
g, d, a_T = planet['mars']

# Parameter related to flying ball (constant)
m = 0.6
r = 0.12
Cd = 0.54
A = pi*r**2

# Initial condition @(x, y) position
x0, y0 = 0, 2
# Initial condition @velocity
vel0 = 10.0     

# Angle of projectile
a0 = 80

# Initial velocity
u0 = vel0 * cos(a0*pi/180)
v0 = vel0 * sin(a0*pi/180)

# Time domain: 0 - t_T
t_T = a_T    # let it fly..

# Sample size
n_pde = 10000
n_ic = 1
t = jnp.linspace(0, t_T, n_pde).reshape(-1, 1)

# Loss Function will be used as the fitness  (f=-L)
def loss(prediction, inputs):
    # essentially the new prediction here
    t = inputs[:,0:1].reshape(-1,1)
    x, y, x_t, y_t, x_tt, y_tt = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3], prediction[:,3:4], prediction[:,4:5], prediction[:,5:6]
    # ground truth
    x_true = f_x(t)
    y_true = f_y(t)  
    _ic = jnp.where((jnp.equal(t, 0)), 1, 0)
    # initial conditions (which define the problem)
    ic_1 = jnp.sum(jnp.square((x_true-x)*_ic))/jnp.sum(_ic)
    ic_2 = jnp.sum(jnp.square((y_true-y)*_ic))/jnp.sum(_ic)
    ic_3 = jnp.sum(jnp.square((x_t-u0)*_ic))/jnp.sum(_ic)
    ic_4 = jnp.sum(jnp.square((y_t-v0)*_ic))/jnp.sum(_ic)
    # sum up all initial conditions
    ic_mse = ic_1 + ic_2 + ic_3 + ic_4
    # consider drag effect
    V = jnp.sqrt( (x_t)**2 + (y_t)**2 )
    C = 0.5*d*Cd*A/m
    R = C*V
    # PDE (physics laws):
    pde_x = x_tt + R*x_t
    pde_y = y_tt + R*y_t + g
    # exclude BC points
    _pde = jnp.where((jnp.equal(t, 0)), 0, 1)
    pde_x = pde_x*_pde
    pde_y = pde_y*_pde        
    pde_residuals_x = jnp.sum(jnp.square(pde_x))/jnp.sum(_pde)
    pde_residuals_y = jnp.sum(jnp.square(pde_y))/jnp.sum(_pde)
    pde_mse = pde_residuals_x + pde_residuals_y
    loss = pde_mse + ic_mse
    return loss

# Task
class Projectile(VectorizedTask):
    """PINN for 2D projectile equation.
    We model the regression as a one-step task, i.e., 'Diffusion.reset' returns a batch of data to the agent,
    the agent outputs predictions, `Diffusion.step` returns the reward loss) and terminates the rollout."""
    def __init__(self):
        self.max_steps = 1
        # Input shape is 1
        self.obs_shape = tuple([1, ])
        # Output shape is 2
        self.act_shape = tuple([2, ])
        # PDE data
        data_pde = t.reshape(-1,1)
        x, y = f_x(data_pde), f_y(data_pde)
        # samples, features = data_pde.shape
        labels_pde = jnp.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
        def reset_fn(key):
            batch_data, batch_labels = data_pde, labels_pde
            return State(obs=batch_data, labels=batch_labels)
        # We use jax.vmap for auto-vectorization.
        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        # step_fn needs to be changed
        def step_fn(state, action):
            reward = -loss(action, state.obs)
            return state, reward, jnp.ones(())
        # We use jax.vmap for auto-vectorization.
        self._step_fn = jax.jit(jax.vmap(step_fn))
    
    # Returns a batch of data to the agent, the agent outputs predictions
    def reset(self, key):
        return self._reset_fn(key)
    
    # Returns the fitness loss
    def step(self, state, action):
        return self._step_fn(state, action)


# PINN architecture
# Number of nodes per hidden layer
n_nodes = 8

class PINNs(nn.Module):
    def setup(self):
        # hidden layers
        self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False),
                       nn.tanh,
                       nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh]
        # split layers
        self.splitx = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh,
                       nn.Dense(1, use_bias=False)]
        self.splity = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh,
                       nn.Dense(1, use_bias=False)]
        
    @nn.compact
    def __call__(self, inputs):
        # Split the variables if necessary
        t = inputs
        def get_xy(t):
            for i, lyr in enumerate(self.layers):
                t = lyr(t)
            # split layers
            hidden_x, hidden_y = self.splitx[0](t), self.splity[0](t)
            hidden_x, hidden_y = self.splitx[1](hidden_x), self.splity[1](hidden_y)
            # output layers
            x, y = self.splitx[2](hidden_x), self.splity[2](hidden_y)
            return x, y
        # Obtain x, y
        x, y = get_xy(t)
        # Obtain deriviatives x_t, y_t
        def get_xy_t(get_xy, t):
            x_t, y_t = jacfwd(get_xy)(t)
            return x_t, y_t
        xy_t_vmap = vmap(get_xy_t, in_axes=(None, 0))
        x_t, y_t = xy_t_vmap(get_xy, t)
        x_t, y_t = x_t.reshape(-1,1), y_t.reshape(-1, 1)
        # Obtain deriviatives x_tt, y_tt
        def get_xy_tt(get_xy, t):
            x_tt, y_tt = hessian(get_xy)(t)
            return x_tt, y_tt
        xy_tt_vmap = vmap(get_xy_tt, in_axes=(None, 0))
        x_tt, y_tt = xy_tt_vmap(get_xy, t)
        x_tt, y_tt = x_tt.reshape(-1,1), y_tt.reshape(-1, 1)
        # Obtain outputs
        action = jnp.hstack([x, y, x_t, y_t, x_tt, y_tt])   
        return action

# Policy
class PINNsPolicy(PolicyNetwork):
    # PINN for the diffusion task.
    def __init__(self):
        model = PINNs()
        key1, key2 = random.split(random.PRNGKey(seed))
        a = random.normal(key1, [1,1]) # Dummy input
        params = model.init(key2, a) # Initialization call
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states

    
# Initialize task & policy
seed = 0
train_task = Projectile()
policy = PINNsPolicy()
sim_mgr = SimManager(n_repeats=1, test_n_repeats=1, pop_size=0, n_evaluations=1,
                     policy_net=policy, train_vec_task=train_task, valid_vec_task=train_task,
                     seed=seed)

# Function takes in the parameters of the neural network to produce the fitness/loss.
# This function will ultimately be imported into our solver document.
def get_fitness(samples):
    scores, _ = sim_mgr.eval_params(params=samples, test=False)
    return scores