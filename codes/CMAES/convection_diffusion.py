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
# Choose GPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Functions and data
# Contains the train input and its labels
@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray

# Function to generate analytical solution
def eval_u(x):
    u = (1. - jnp.exp(Pe*x/L)) / (1. - jnp.exp(Pe))
    return u

# Choose v (PDE parameter)
v = 6
k = 1.
L = 1.
Pe = v*L/k

# Domain Boundary
x_l, x_u = 0, L

# Sample size
n = 10000

# Loss Function will be used as the fitness  (f=-L)
def loss(prediction, inputs):
    # Essentially the new prediction here
    x = inputs[:,0:1].reshape(-1,1)
    u, u_x, u_xx = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3]
    # Ground truth
    t_u = eval_u(x)
    _bc = jnp.where((jnp.equal(x, x_l)|jnp.equal(x, x_u)), 1, 0)
    # _bc = ( jnp.equal(x, x_l) | jnp.equal(x, x_u) )
    bc_mse = jnp.sum(jnp.square((t_u-u)*_bc))/jnp.sum(_bc)
    # PDE (physics laws): v*u_x = k*u_xx
    pde = v*u_x - k*u_xx
    # Exclude BC points
    _pde = jnp.where((jnp.equal(x, x_l)|jnp.equal(x, x_u)), 0, 1)
    pde = pde*_pde
    pde_mse = jnp.sum(jnp.square(pde))/jnp.sum(_pde)
    loss = pde_mse + bc_mse
    return loss

# Task
class Diffusion(VectorizedTask):
    """PINN for 1D convection-diffusion equation.
    We model the regression as a one-step task, i.e., 'Diffusion.reset' returns a batch of data to the agent,
    the agent outputs predictions, `Diffusion.step` returns the reward loss) and terminates the rollout."""
    def __init__(self):
        self.max_steps = 1
        # Input shape is 1
        self.obs_shape = tuple([1, ])
        # Output shape is 1
        self.act_shape = tuple([1, ])
        # PDE data
        x = jnp.linspace(x_l, x_u, n)
        data_pde = x.reshape(-1, 1)
        labels_pde = jnp.reshape(eval_u(x),(-1, 1))
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
n_nodes = 10

class PINNs(nn.Module):
    def setup(self):
        self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh,
                       nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh,
                       nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                       nn.tanh,
                       nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)]

    @nn.compact
    def __call__(self, inputs):
        # Split the variables if necessary
        x = inputs
        def get_u(x):
            u = x
            for i, lyr in enumerate(self.layers):
                u = lyr(u)
            return u
        # Obtain u
        u = get_u(x)
        # Obtain u_x
        def get_u_x(get_u, x):
            u_x = jacfwd(get_u)(x)
            return u_x
        u_x_vmap = vmap(get_u_x, in_axes=(None, 0))
        u_x = u_x_vmap(get_u, x).reshape(-1, 1)
        # Obtain u_xx
        def get_u_xx(get_u, x):
            u_xx = hessian(get_u)(x)
            return u_xx
        u_xx_vmap = vmap(get_u_xx, in_axes=(None, 0))
        u_xx = u_xx_vmap(get_u, x).reshape(-1, 1)
        # Obtain outputs
        action = jnp.hstack([u, u_x, u_xx])
        return action

# Policy
class PINNsPolicy(PolicyNetwork):
    # PINN for the diffusion task.
    def __init__(self):
        model = PINNs()
        key1, key2 = random.split(random.PRNGKey(seed))
        a = random.normal(key1, [1,1]) # Dummy input
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
train_task = Diffusion()
policy = PINNsPolicy()
sim_mgr = SimManager(n_repeats=1, test_n_repeats=1, pop_size=0, n_evaluations=1,
                     policy_net=policy, train_vec_task=train_task, valid_vec_task=train_task,
                     seed=seed)

# Function takes in the parameters of the neural network to produce the fitness/loss.
# This function will ultimately be imported into our solver document.
def get_fitness(samples):
    scores, _ = sim_mgr.eval_params(params=samples, test=False)
    return scores
