import gym
import torch
from models import Actor, Critic
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_value_

#discount factor
GAMMA = 0.99
#entropy penalty coefficient
BETA = 0.06
LR = 1e-3
GRAD_CLIP_VALUE = 25.0
#create env
env = gym.make("CartPole-v1")

NUM_ACTIONS = env.action_space.n
NUM_EPISODES = 800
MAX_STEPS = env.spec.max_episode_steps #track so that we don't treat timeouts as terminal states

#make buffers
#buffer = [] 
actor = Actor(4, NUM_ACTIONS)
critic = Critic(4)
a_optimizer = torch.optim.Adam(actor.parameters(), lr=LR)
c_optimizer = torch.optim.Adam(critic.parameters(), lr=LR)
a_scheduler = CosineAnnealingLR(a_optimizer, T_max=NUM_EPISODES*300)
c_scheduler = CosineAnnealingLR(c_optimizer, T_max=NUM_EPISODES*300)
#torch.autograd.detect_anomaly()

episode_rewards = []
ploss = []
closs = []
lrs = []


lrs.append(LR)

for episode in range(NUM_EPISODES):
	state = env.reset()
	done = False
	r = 0
	n_steps = 0
	while not done:
		print(episode)
		print(state)
		action_dist = actor(torch.from_numpy(state).reshape(1, len(state)).float())
		value = critic(torch.from_numpy(state).reshape(1, len(state)).float())
		#sample action from actor output
		print(action_dist)
		dist = torch.distributions.Categorical(probs=action_dist)
		act = dist.sample()

		print(act.detach().data.numpy()[0])
		new_state, reward, done, _ = env.step(act.detach().data.numpy()[0])
		env.render()

		n_steps += 1
		r += reward

		if n_steps >= MAX_STEPS:
			episode_rewards.append(r)

		#calculate advantage:
		#A(s,a) = r_t+1 + GAMMA*V(s_t+1) - V(s_t)
		if not done or n_steps >= MAX_STEPS:
			future_value = critic(torch.from_numpy(new_state).reshape(1, len(new_state)).float())
			advantage = reward + GAMMA*future_value - value
		else:
			advantage = reward - value 
			episode_rewards.append(r)

		print(f'Advantage: {advantage}')

		#calc entropy
		entropy_penalty = dist.entropy()
		
		#calc critic loss: 1/T ||A(s,t)||^2
		critic_loss = .5*torch.pow(advantage, 2).mean()
		c_optimizer.zero_grad()
		critic_loss.backward()
		clip_grad_value_(critic.parameters(), GRAD_CLIP_VALUE)
		c_optimizer.step()

		policy_loss = (advantage.detach()*(-dist.log_prob(act))) - BETA*entropy_penalty
		a_optimizer.zero_grad()
		policy_loss.backward()
		clip_grad_value_(actor.parameters(), GRAD_CLIP_VALUE)
		a_optimizer.step()

		

		state = new_state

		#losses.append(n_steps)#loss.item())
		ploss.append(policy_loss.item())
		closs.append(critic_loss.item())

		print('-----------------')

	#lr scheduler steps	
	a_scheduler.step()
	c_scheduler.step()

	lrs.append(a_scheduler.get_last_lr()[0])



#plot episode rewards, loss, policy loss, critic loss
fig, axs = plt.subplots(2,2)
fig.tight_layout()

axs[0,0].scatter(range(NUM_EPISODES), episode_rewards, s=2)
axs[0,0].set_xlabel("Episode")
axs[0,0].set_ylabel("Net Reward")
axs[0,0].set_title("Net Reward over each episode")


axs[0,1].plot(range(len(lrs)), lrs)
axs[0,1].set_xlabel("Episode")
axs[0,1].set_ylabel("LR")
axs[0,1].set_title("Learning Rates over each Episode")


axs[1,0].plot(range(len(ploss)), ploss)
axs[1,0].set_xlabel("Episode")
axs[1,0].set_ylabel("Policy Loss")
axs[1,0].set_title("Policy Loss over each timestep")


axs[1,1].plot(range(len(closs)), closs)
axs[1,1].set_xlabel("Episode")
axs[1,1].set_ylabel("Critic Loss")
axs[1,1].set_title("Critic Loss over each timestep")


plt.show()