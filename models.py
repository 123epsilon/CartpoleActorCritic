import torch
from torch import nn

class Actor(nn.Module):
	def __init__(self, state_dims, num_actions, hidden_dim=64):
		super(Actor,self).__init__()

		self.actor = nn.Sequential(
				nn.Linear(state_dims, hidden_dim),
				nn.LeakyReLU(),
				#nn.Linear(hidden_dim, hidden_dim),
				#nn.LeakyReLU(),
				#nn.Linear(hidden_dim, hidden_dim),
				#nn.LeakyReLU(),
				nn.Linear(int(hidden_dim), num_actions),
				nn.Softmax()
			)

	def forward(self, x):
		return self.actor(x)

class Critic(nn.Module):
	def __init__(self, state_dims, hidden_dim=64):
		super(Critic,self).__init__()

		self.critic = nn.Sequential(
				nn.Linear(state_dims, hidden_dim),
				nn.LeakyReLU(),
				#nn.Linear(hidden_dim, hidden_dim),
				#nn.LeakyReLU(),
				#nn.Linear(hidden_dim, hidden_dim),
				#nn.LeakyReLU(),
				nn.Linear(int(hidden_dim), 1)
			)

	def forward(self, x):
		return self.critic(x)