# Import libraries
import pandas as pd
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
# Choose GPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Functions and data
# Contains the train input and its labels
@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray

# Function to generate analytical solution of IC
def f_ic(x):
    return m*jnp.exp(-(k*x)**2)

# Choose v (PDE parameter)
vis = 0.02
c = 1

# Initial condition
k = 2
m = 10

# Spatial domain
x_l, x_u = -1.5, 4.5

# Time domain: 0 - t_T
t_T = 2.0 

# Read data
sim = pd.read_csv('linear.csv')
sim = sim[sim.x <= 4.5]
batch_X = np.vstack([sim.x.values, sim.t.values]).T
batch_y = sim[['u']].values

# Loss Function will be used as the fitness  (f=-L)
def loss(prediction, inputs):
    # Essentially the new prediction here
    x, t = inputs[:,0:1].reshape(-1,1), inputs[:,1:2].reshape(-1,1)    
    u, u_x, u_xx, u_t = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3], prediction[:,3:4]
    # Ground truth (IC)
    t_u = f_ic(x) 
    _ic = jnp.where((jnp.equal(t, 0)), 1, 0)
    ic_mse = jnp.sum(jnp.square((t_u-u)*_ic))/jnp.sum(_ic)
    # PDE (physics laws): u_t + c*u_x - vis*u_xx = 0   
    pde = u_t + c*u_x - vis*u_xx
    # Exclude IC points
    _pde = jnp.where((jnp.equal(t, 0)), 0, 1)
    pde = pde*_pde
    pde_mse = jnp.sum(jnp.square(pde))/jnp.sum(_pde)
    loss = pde_mse + ic_mse
    return loss

# Task
class linear(VectorizedTask):
    """PINN for 1D transient linearized Burgers' equation.
    We model the regression as a one-step task, i.e., 'linear.reset' returns a batch of data to the agent,
    the agent outputs predictions, `linear.step` returns the reward loss and terminates the rollout."""
    def __init__(self):
        self.max_steps = 1
        # Input shape is 2
        self.obs_shape = tuple([2, ])
        # Output shape is 1
        self.act_shape = tuple([1, ])
        # PDE data
        data_pde, labels_pde = batch_X, batch_y        
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
node = 10

class PINNs(nn.Module):
    def setup(self):
        self.layers = [nn.Dense(node, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh,
                       nn.Dense(node, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh,
                       nn.Dense(node, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh,
                       nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)]

    @nn.compact
    def __call__(self, inputs):
        # Split the variables if necessary
        x, t = inputs[:,0:1], inputs[:,1:2]
        def get_u(x, t):
            u = jnp.hstack([x, t])
            for i, lyr in enumerate(self.layers):
                u = lyr(u)
            return u
        # Obtain u
        u = get_u(x, t)
        # Obtain u_t
        def get_u_t(get_u, x, t):
            u_t = jacfwd(get_u, 1)(x, t)
            return u_t
        u_t_vmap = vmap(get_u_t, in_axes=(None, 0, 0))
        u_t = u_t_vmap(get_u, x, t).reshape(-1,1) 
        # Obtain u_x
        def get_u_x(get_u, x, t):
            u_x = jacfwd(get_u)(x, t)
            return u_x
        u_x_vmap = vmap(get_u_x, in_axes=(None, 0, 0))
        u_x = u_x_vmap(get_u, x, t).reshape(-1,1)
        # Obtain u_xx
        def get_u_xx(get_u, x, t):
            u_xx = hessian(get_u)(x, t)
            return u_xx
        u_xx_vmap = vmap(get_u_xx, in_axes=(None, 0, 0))
        u_xx = u_xx_vmap(get_u, x, t).reshape(-1,1)
        # Obtain outputs
        action = jnp.hstack([u, u_x, u_xx, u_t])
        return action
    
# Policy
class PINNsPolicy(PolicyNetwork):
    # PINN for the diffusion task.
    def __init__(self):
        model = PINNs()
        key1, key2 = random.split(random.PRNGKey(seed))
        a = random.normal(key1, [1,2]) # Dummy input
        params = model.init(key2, a) # Initialization call
        # Return a function that formats the parameters into a correct format.
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
train_task = linear()
policy = PINNsPolicy()
sim_mgr = SimManager(n_repeats=1, test_n_repeats=1, pop_size=0, n_evaluations=1,
                     policy_net=policy, train_vec_task=train_task, valid_vec_task=train_task,
                     seed=seed)

# Function takes in the parameters of the neural network to produce the fitness/loss.
# This function will ultimately be imported into our solver document.
def get_fitness(samples):
    scores, _ = sim_mgr.eval_params(params=samples, test=False)
    return scores
