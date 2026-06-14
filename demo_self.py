import gymnasium as gym
import gymnasium_stag_hunt
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import time
import random
import pickle

DEVICE = "cpu"

# Seeding
torch.manual_seed(420)
np.random.seed(69)
random.seed(67)

def greedy_action(probs: torch.tensor) -> int:
    return probs.argmax().item()

class Actor(nn.Module):
    def __init__(self, time_frames=1, actions=5, hidden_channels=64, game='Hunt'):
        super().__init__()
        self.input_layer = nn.Linear(10, hidden_channels)
        self.hidden_layer = nn.Linear(hidden_channels, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, actions)
        self.ReLU = nn.ReLU()

    def forward(self, x:torch.tensor):
        x = self.input_layer(x)
        x = self.ReLU(x)
        x = self.hidden_layer(x)
        x = self.ReLU(x)
        x = self.output_layer(x)
        return x


# environment
game = 'Hunt'
env = gym.make("StagHunt-"+game+"-v0", obs_type="coords", max_episode_steps=200, stag_follows=False, load_renderer=True)

actor = torch.load("self.pt", weights_only=False)
actor.eval()
(obs1, obs2), _ = env.reset()
env.render()
time.sleep(1)
for _ in range(200):
    obs1 = torch.FloatTensor(obs1).to(DEVICE)
    obs2 = torch.FloatTensor(obs2).to(DEVICE)
    probs1 = actor(obs1)
    probs2 = actor(obs2)
    a1 = greedy_action(probs1)
    a2 = greedy_action(probs2)
    (obs1, obs2), (r1, r2), terminated, truncated, info = env.step((a1, a2))
    env.render()
    time.sleep(1)
