import gymnasium as gym
import gymnasium_stag_hunt
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import time
import random
# Seeding
torch.manual_seed(420)
np.random.seed(69)
random.seed(67)
# Constants

MAX_PLANTS = 2
PLANTS_REWARD = 1
HC = 64
LR_ACTOR = 0.0005
LR_CRITIC = 0.001
MAX_STEPS = 200
EPISODES = 25000
GAME_OBS = 2 * 2 + 2 * 1 + 2 * MAX_PLANTS # 2 coords per agent (2 agents), 2 coords per stag (1 stag), 2coords per plant (MAX_PLANTS plants)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# environment
game = 'Hunt'
env = gym.make("StagHunt-"+game+"-v0", obs_type="coords", max_episode_steps=MAX_STEPS, forage_quantity=MAX_PLANTS, forage_reward=PLANTS_REWARD, stag_follows=False)

class Actor(nn.Module):
    def __init__(self, time_frames=1, actions=5, hidden_channels=HC, game='Hunt'):
        super().__init__()
        self.input_layer = nn.Linear(time_frames * GAME_OBS, hidden_channels)
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

class Critic(nn.Module):
    def __init__(self, time_frames=1, hidden_channels=HC, game='Hunt'):
        super().__init__()
        self.input_layer = nn.Linear(time_frames * GAME_OBS, hidden_channels)
        self.hidden_layer = nn.Linear(hidden_channels, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, 1)
        self.ReLU = nn.ReLU()

    def forward(self, x:torch.tensor):
        x = self.input_layer(x)
        x = self.ReLU(x)
        x = self.hidden_layer(x)
        x = self.ReLU(x)
        x = self.output_layer(x)
        return x


# Util

def greedy_action(probs: torch.tensor) -> int:
    return probs.argmax().item()

def sample_action(probs: torch.tensor) -> int:
    categorical = Categorical(probs=probs)
    sample = categorical.sample()
    log_prob = categorical.log_prob(sample)
    sample = sample.item()
    return (sample, log_prob)


# Training functions

def hunt_train_loop(
    env, # The gym env
    actor1, # The agent1 actor model
    actor2, # The agent2 actor model
    critic, # The critic  model
    actor1_optimizer, # The torch optimizer for the first actor model
    actor2_optimizer, # The torch optimizer for the second actor model
    critic_optimizer, # The torch optimizer for the critic model
    n_episodes: int = EPISODES, # The episodes to run (episodes ~ # games, not # steps)
    gamma: float = 0.99, # The discount for the advantage
    dist_coeff: float = 0.1, # The distance penalty coefficient 
    stag_rew: float = 5.00, # The expected reward when successfully capturing a stag
    log_every: int = 100, # How often to log training
):
    # The logs, will be used later
    logs = {
            "episode":            [],
            "total_reward_1":     [],   # sum of rewards for agent 1 over the episode
            "total_reward_2":     [],   # sum of rewards for agent 2 over the episode
            "mean_reward_1":      [],   # avg reward per step for agent 1
            "mean_reward_2":      [],   # avg reward per step for agent 2
            "actor1_loss":        [],   # mean actor2 loss over the episode
            "actor2_loss":        [],   # mean actor2 loss over the episode
            "critic_loss":        [],   # mean critic loss over the episode
            "stag_catches":       [],   # amount of stags catched
            "plants_harvested_1": [],   # plants harvested by agent 1
            "plants_harvested_2": [],   # plants harvested by agent 2
            "episode_length":     [],
    }



    for ep in range(n_episodes):
 
        # reset
        (obs1, obs2), _ = env.reset()
        obs1 = torch.FloatTensor(obs1).to(DEVICE)
        obs2 = torch.FloatTensor(obs2).to(DEVICE)
 
        ep_rewards_1      = []
        ep_rewards_2      = []
        ep_actor1_losses = []
        ep_actor2_losses  = []
        ep_critic_losses  = []
        stag_catches      = 0
        step              = 0
        harvested1        = 0
        harvested2        = 0
        done              = False
 
        # episode loop
        while not done:
 
            # select action (probabilistic since we are training)
            probs1 = torch.softmax(actor1(obs1),dim=-1)
            probs2 = torch.softmax(actor2(obs2),dim=-1)

            # ADD THIS
            if step == 0 and ep % 500 == 0 and ep != 0:
                print(f"Ep {ep} | probs1: {probs1.detach().cpu().numpy().round(3)}")
                print(f"Ep {ep} | stag_catches last 500: {sum(logs['stag_catches'][-500:])}")

            a1, log_prob1 = sample_action(probs1)
            a2, log_prob2 = sample_action(probs2)
 
            # step
            (obs1_next, obs2_next), (r1, r2), terminated, truncated, info = \
                env.step((a1, a2))
 
            done = terminated or truncated
 
            obs1_next = torch.FloatTensor(obs1_next).to(DEVICE)
            obs2_next = torch.FloatTensor(obs2_next).to(DEVICE)
 
            # shared reward
            r_shared = (r1 + r2) / (2*stag_rew) # scale down to -1 ~ +1
            a1x, a1y = obs1_next[0].item(), obs1_next[1].item()
            a2x, a2y = obs1_next[2].item(), obs1_next[3].item()
            sx,  sy  = obs1_next[4].item(), obs1_next[5].item()

            dist1 = abs(a1x - sx) + abs(a1y - sy)
            dist2 = abs(a2x - sx) + abs(a2y - sy)

            # Far from stag = bad
            r_shared -= dist_coeff * (dist1 + dist2) / (2 * 5)

            ep_rewards_1.append(r1)
            ep_rewards_2.append(r2)
 
            # track cooperation
            # # How? If reward shared is roughly equal to stag reward,
            # # count the step towards cooperation
            if abs(r1 + r2 - 2*stag_rew) < 1e-2:
                stag_catches += 1
            
            # Check for harvests
            if abs(r1 - PLANTS_REWARD) < 1e-2:
                harvested1 += 1
            
            if abs(r2 - PLANTS_REWARD) < 1e-2:
                harvested2 += 1
            
            # critic update
            # # Use only agent1 obs, as they are equal between agents
            v = critic(obs1)
 
            with torch.no_grad(): # no gradient when predicting next critic value!!!
                v_next    = critic(obs1_next)
                td_target = r_shared + gamma * v_next * (1.0 - float(done))
 
            advantage    = td_target - v
            critic_loss  = advantage.pow(2) # MSE
 
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
 
            # actor update
            # Both agents contribute gradients to the shared network
            adv_detached = advantage.detach()


            entropy_coef = max(0.001, 0.05 * (0.9998 ** ep))
            # Add entropy to enhance exploration
            entropy1 = -(probs1 * torch.log(probs1 + 1e-8)).sum()

            actor1_loss = -(log_prob1) * adv_detached - (entropy_coef * entropy1)
 
            actor1_optimizer.zero_grad()
            actor1_loss.backward()
            actor1_optimizer.step()

            entropy2 = -(probs2 * torch.log(probs2 + 1e-8)).sum()

            actor2_loss = -(log_prob2) * adv_detached - (entropy_coef * entropy2)
            
            actor2_optimizer.zero_grad()
            actor2_loss.backward()
            actor2_optimizer.step()
 
            # --- bookkeeping ---
            ep_actor1_losses.append(actor1_loss.item())
            ep_actor2_losses.append(actor2_loss.item())
            ep_critic_losses.append(critic_loss.item())
 
            obs1 = obs1_next
            obs2 = obs2_next
            step += 1
 
        # episode logging 
        total_reward_1   = float(np.sum(ep_rewards_1))
        total_reward_2   = float(np.sum(ep_rewards_2))
        mean_reward_1    = total_reward_1/step
        mean_reward_2    = total_reward_2/step
        mean_actor1_loss = float(np.mean(np.abs(ep_actor1_losses)))
        mean_actor2_loss = float(np.mean(np.abs(ep_actor2_losses)))
        mean_critic_loss = float(np.mean(ep_critic_losses))
 
        logs["episode"].append(ep)
        logs["total_reward_1"].append(total_reward_1)
        logs["total_reward_2"].append(total_reward_2)
        logs["mean_reward_1"].append(mean_reward_1)
        logs["mean_reward_2"].append(mean_reward_2)
        logs["actor1_loss"].append(mean_actor1_loss)
        logs["actor2_loss"].append(mean_actor2_loss)
        logs["critic_loss"].append(mean_critic_loss)
        logs["stag_catches"].append(stag_catches)
        logs["plants_harvested_1"].append(harvested1)
        logs["plants_harvested_2"].append(harvested2)
        logs["episode_length"].append(step)
 
        if ep % log_every == 0:
            print(
                f"Ep {ep:5d} | "
                f"TotRew1: {total_reward_1:7.2f} | "
                f"TotRew2: {total_reward_2:7.2f} | "
                f"StagCatch: {stag_catches:.2f} | "
                f"Harvested1: {harvested1} | "
                f"Harvested2: {harvested2} | "
                f"Actor1L: {mean_actor1_loss:.4f} | "
                f"Actor2L: {mean_actor2_loss:.4f} | "
                f"CriticL: {mean_critic_loss:.4f} | "
                f"Steps: {step}"
            )
 
    return logs


def single_agent_hunt_train_loop(
        env, # The gym env
        actor, # The actor model
        critic, # The critic model
        actor_optimizer, # The torch optimizer for the actor model
        critic_optimizer, # The torch optimizer for the critic model
        n_episodes: int = EPISODES, # The episodes to run (episodes ~ # games, not # steps)
        gamma: float = 0.99, # The discount for the advantage
        dist_coeff: float = 0.1, # The distance penalty coefficient 
        stag_rew: float = 5.00, # The expected reward when successfully capturing a stag
        log_every: int = 100, # How often to log training
    ):
    # The logs, will be used later
    logs = {
            "episode":            [],
            "total_reward_1":     [],   # sum of rewards for agent 1 over the episode
            "mean_reward_1":      [],   # avg reward per step for agent 1
            "actor_loss":         [],   # mean actor loss over the episode
            "critic_loss":        [],   # mean critic loss over the episode
            "stag_catches":       [],   # amount of stags catched
            "plants_harvested_1": [],   # plants harvested by agent 1
            "episode_length":     [],
    }

    for ep in range(n_episodes):
 
        # reset
        obs1, _ = env.reset()
        obs1 = torch.FloatTensor(obs1).to(DEVICE)
 
        ep_rewards_1      = []
        ep_actor_losses   = []
        ep_critic_losses  = []
        stag_catches      = 0     # how many times the stags gets caught
        step              = 0
        harvested1        = 0
        done              = False
 
        # episode loop
        while not done:
 
            # select action (probabilistic since we are training)
            probs1 = torch.softmax(actor(obs1),dim=-1)
 
            a1, log_prob1 = sample_action(probs1)
 
            # step
            obs1_next, r1, terminated, truncated, info = env.step(a1)
 
            done = terminated or truncated
 
            obs1_next = torch.FloatTensor(obs1_next).to(DEVICE)
            a1x, a1y = obs1_next[0].item(), obs1_next[1].item()
            a2x, a2y = obs1_next[2].item(), obs1_next[3].item()
            sx,  sy  = obs1_next[4].item(), obs1_next[5].item()
            
            dist1 = abs(a1x - sx) + abs(a1y - sy)
            dist2 = abs(a2x - sx) + abs(a2y - sy)

            # shared reward
            r_scaled = r1 / stag_rew # Scale down to -1 ~ +1
            ep_rewards_1.append(r1)
            
            r_scaled -= dist_coeff * (dist1 + dist2) / (2 * 5)
 
            # track cooperation
            # # How? If reward shared is roughly equal to stag reward,
            # # count the step towards cooperation
            if abs(r1 - stag_rew) < 1e-2:
                stag_catches += 1
            
            if abs(r1 - 1.0) < 1e-2:
                harvested1 += 1
 
            # critic update
            # # Use only agent1 obs, as they are equivalent between agents
            v = critic(obs1)
 
            with torch.no_grad(): # no gradient when predicting next critic value!!!
                v_next    = critic(obs1_next)
                td_target = r_scaled + gamma * v_next * (1.0 - float(done))
 
            advantage    = td_target - v
            critic_loss  = advantage.pow(2) # MSE
 
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
 
            # actor update
            adv_detached = advantage.detach()

            entropy_coef = max(0.001, 0.05 * (0.9998 ** ep))
            # Add entropy to enhance exploration
            entropy = -(probs1 * torch.log(probs1 + 1e-8)).sum() 

            actor_loss = -(log_prob1) * adv_detached - (entropy_coef * entropy)
 
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            # logging
            ep_actor_losses.append(actor_loss.item())
            ep_critic_losses.append(critic_loss.item())
 
            obs1 = obs1_next
            step += 1
 
        # episode logging 
        total_reward_1   = float(np.sum(ep_rewards_1))
        mean_reward_1    = total_reward_1/step
        mean_actor_loss  = float(np.mean(np.abs(ep_actor_losses)))
        mean_critic_loss = float(np.mean(ep_critic_losses))
 
        logs["episode"].append(ep)
        logs["total_reward_1"].append(total_reward_1)
        logs["mean_reward_1"].append(mean_reward_1)
        logs["stag_catches"].append(stag_catches)
        logs["plants_harvested_1"].append(harvested1)
        logs["actor_loss"].append(mean_actor_loss)
        logs["critic_loss"].append(mean_critic_loss)
        logs["episode_length"].append(step)
 
        if ep % log_every == 0:
            print(
                f"Ep {ep:5d} | "
                f"TotRew1: {total_reward_1:7.2f} | "
                f"StagCatch: {stag_catches:.2f} | "
                f"Harvested1: {harvested1} | "
                f"ActorL: {mean_actor_loss:.4f} | "
                f"CriticL: {mean_critic_loss:.4f} | "
                f"Steps: {step}"
            )
 
    return logs


# Test functions

def hunt_eval_loop(env, actor1, actor2, n_episodes=100, log_every=10):
    logs = {
            "episode":                [],
            "total_reward_cumul":     [],   # sum of shared rewards over the episode
            "mean_reward_cumul":      [],   # avg shared reward per step
            "stag_catches":           [],   # amount of stags catched
            "plants_harvested_cumul": [],   # plants harvested by two agents
            "episode_length":         [],
    }
    for ep in range(n_episodes):
 
        # reset
        (obs1, obs2), _ = env.reset()
        obs1 = torch.FloatTensor(obs1).to(DEVICE)
        obs2 = torch.FloatTensor(obs2).to(DEVICE)
 
        ep_rewards        = []
        stag_catches      = 0     # how many times the stags gets caught
        step              = 0
        harvested         = 0
        done              = False
 
        # episode loop
        while not done:
 
            # select action (since we are testing, take highest)
            probs1 = torch.softmax(actor1(obs1), dim=-1) 
            probs2 = torch.softmax(actor2(obs2), dim=-1)
 
            a1 = greedy_action(probs1)
            a2 = greedy_action(probs2)
 
            # step
            (obs1_next, obs2_next), (r1, r2), terminated, truncated, info = \
                env.step((a1, a2))
 
            done = terminated or truncated
 
            obs1_next = torch.FloatTensor(obs1_next).to(DEVICE)
            obs2_next = torch.FloatTensor(obs2_next).to(DEVICE)
 
            # shared reward
            r_shared = (r1 + r2) / 2.0
            ep_rewards.append(r_shared)
 
            # track cooperation
            # # How? If reward shared is roughly equal to stag reward,
            # # count the step towards cooperation
            if abs(r_shared - 5.0) < 1e-2:
                stag_catches += 1
            
            # Plants harvested
            if abs(r1 - 1.0) < 1e-2:
                harvested += 1
            
            if abs(r2 - 1.0) < 1e-2:
                harvested += 1
 
            obs1 = obs1_next
            obs2 = obs2_next
            step += 1
 
        # episode logging 
        total_reward     = float(np.sum(ep_rewards))
        mean_reward      = float(np.mean(ep_rewards))
 
        logs["episode"].append(ep)
        logs["total_reward_cumul"].append(total_reward)
        logs["mean_reward_cumul"].append(mean_reward)
        logs["stag_catches"].append(stag_catches)
        logs["plants_harvested_cumul"].append(harvested)
        logs["episode_length"].append(step)
 
        if ep % log_every == 0:
            print(
                f"Ep {ep:5d} | "
                f"TotRew: {total_reward:7.2f} | "
                f"StagCatch: {stag_catches:.2f} | "
                f"Harvested: {harvested} | "
                f"Steps: {step}"
            )
 
    return logs


def single_agent_hunt_eval_loop(env, actor, n_episodes=100, log_every=10):
    logs = {
            "episode":            [],
            "total_reward_1":     [],   # sum of shared rewards over the episode
            "mean_reward_1":      [],   # avg shared reward per step
            "stag_catches":    [],   # amount of steps to catch stag
            "plants_harvested_1": [],   # amount of plants harvested
            "episode_length":     [],
    }
    for ep in range(n_episodes):
 
        # reset
        obs1, _ = env.reset()
        obs1 = torch.FloatTensor(obs1).to(DEVICE)
 
        ep_rewards        = []
        stag_catches      = 0     # how many times the stags gets caught
        step              = 0
        harvested         = 0
        done              = False
 
        # episode loop
        while not done:
 
            # select action (probabilistic since we are training)
            probs1 = torch.softmax(actor(obs1), dim=-1)
 
            a1 = greedy_action(probs1)
 
            # step
            obs1_next, r1, terminated, truncated, info = \
                env.step(a1)
 
            done = terminated or truncated
 
            obs1_next = torch.FloatTensor(obs1_next).to(DEVICE)

            # append reward
            ep_rewards.append(r1)
 
            # track cooperation
            # # How? If reward shared is roughly equal to stag reward,
            # # count the step towards cooperation
            if abs(r1 - 5.0) < 1e-2:
                stag_catches += 1
            # plants harvested
            if abs(r1 - 1.0) < 1e-2:
                harvested += 1
 
            obs1 = obs1_next
            step += 1
 
        # episode logging
        total_reward     = float(np.sum(ep_rewards))
        mean_reward      = float(np.mean(ep_rewards))
 
        logs["episode"].append(ep)
        logs["total_reward_1"].append(total_reward)
        logs["mean_reward_1"].append(mean_reward)
        logs["stag_catches"].append(stag_catches)
        logs["plants_harvested_1"].append(harvested)
        logs["episode_length"].append(step)
 
        if ep % log_every == 0:
            print(
                f"Ep {ep:5d} | "
                f"TotRew: {total_reward:7.2f} | "
                f"StagCatch: {stag_catches:.2f} | "
                f"Harvested: {harvested} | "
                f"Steps: {step}"
            )
 
    return logs


# Plotting functions

def plot_logs(logs: dict, window: int = 25, file_name="training_curves.png"):
    """
    Plot training curves with a rolling average overlay.
    All keys in `logs` except 'episode' are plotted.
    """
 
    def smooth(values, w):
        if len(values) < w:
            return np.array(values)
        kernel = np.ones(w) / w
        return np.convolve(values, kernel, mode="valid")
 
    keys = [k for k in logs if k != "episode" and k != "episode_length"]
    n    = len(keys)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes      = axes.flatten()
 
    episodes = np.array(logs["episode"])
 
    titles = {
        "total_reward_1":         "Total Reward per Episode - agent 1",
        "total_reward_2":         "Total Reward per Episode - agent 2",
        "total_reward_cumul":     "Total avg rewards between 2 agents",
        "mean_reward_1":          "Mean Step Reward - agent 1",
        "mean_reward_2":          "Mean Step Reward - agent 2",
        "mean_reward_cumul":      "Mean Step Reward between 2 agents",
        "actor_loss":             "Actor Loss",
        "actor1_loss":            "Actor 1 Loss",
        "actor2_loss":            "Actor 2 Loss",
        "critic_loss":            "Critic Loss",
        "stag_catches":           "Stags caught during the episode",
        "plants_harvested_1":     "Plants harvested by agent 1",
        "plants_harvested_2":     "Plants harvested by agent 2",
        "plants_harvested_cumul": "Plants harvested by both agents"
    }
    colors = {
        "total_reward_1":          "steelblue",
        "total_reward_2":          "firebrick",
        "total_reward_cumul":      "steelblue",
        "mean_reward_1":           "cornflowerblue",
        "mean_reward_2":           "lightcoral",
        "mean_reward_cumul":       "cornflowerblue",
        "actor_loss":              "tomato",
        "actor1_loss":             "tomato",
        "actor2_loss":             "tomato",
        "critic_loss":             "darkorange",
        "stag_catches":            "seagreen",
        "plants_harvested_1":      "aquamarine",
        "plants_harvested_2":      "orchid",
        "plants_harvested_cumul":  "aquamarine",
    }
 
    for ax, key in zip(axes, keys):
        values   = np.array(logs[key])
        color    = colors.get(key, "gray")
        smoothed = smooth(values, window)
 
        ax.plot(episodes, values, alpha=0.25, color=color, linewidth=1)
        ax.plot(
            episodes[window - 1:] if len(episodes) >= window else episodes,
            smoothed,
            color=color,
            linewidth=2,
            label=f"Rolling avg ({window} ep)",
        )
 
        ax.set_title(titles.get(key, key))
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
 
    # hide unused subplots
    for ax in axes[len(keys):]:
        ax.set_visible(False)
 
    fig.suptitle("Stag Hunt — A2C Training Curves", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(file_name, dpi=250, bbox_inches="tight")
    # plt.show()
    print("Plot saved to " + file_name)


# Running

actor1 = Actor(game="Hunt").to(device=DEVICE)
actor2 = Actor(game="Hunt").to(device=DEVICE)
critic = Critic(game="Hunt").to(device=DEVICE)
actor1_optimizer = torch.optim.Adam(params=actor1.parameters(), lr=LR_ACTOR)
actor2_optimizer = torch.optim.Adam(params=actor2.parameters(), lr=LR_ACTOR)
critic_optimizer = torch.optim.Adam(params=critic.parameters(), lr=LR_CRITIC)

actor1.train()
actor2.train()
critic.train()

logs = hunt_train_loop(env, actor1, actor2, critic, actor1_optimizer, actor2_optimizer, critic_optimizer)
actor1.eval()
actor2.eval()

plot_logs(logs, window=100, file_name="base_training_logs.png")

eval_logs = hunt_eval_loop(env, actor1, actor2, log_every=10)

plot_logs(eval_logs, file_name="base_test_logs.png")

env_random = gym.make("StagHunt-"+game+"-v0", obs_type="coords", max_episode_steps=MAX_STEPS, enable_multiagent=False, opponent_policy="random", stag_follows=False)

actor_for_random  =  Actor(game="Hunt").to(device=DEVICE)
critic_for_random = Critic(game="Hunt").to(device=DEVICE)
actor_fr_optim    = torch.optim.Adam(params=  actor_for_random.parameters(), lr=LR_ACTOR)
critic_fr_optim   = torch.optim.Adam(params= critic_for_random.parameters(), lr=LR_CRITIC)

logs_random = single_agent_hunt_train_loop(env_random, actor_for_random, critic_for_random, actor_fr_optim, critic_fr_optim, dist_coeff = 0.0)

plot_logs(logs_random, window=100, file_name="random_training_logs.png")

actor_for_random.eval()

eval_logs_random = single_agent_hunt_eval_loop(env_random, actor_for_random)

plot_logs(eval_logs_random, file_name="random_test_logs.png")

env_pursuit = gym.make("StagHunt-"+game+"-v0", obs_type="coords",max_episode_steps=MAX_STEPS, enable_multiagent=False, opponent_policy="pursuit", stag_follows=False)

actor_for_pursuit  =  Actor(game="Hunt").to(device=DEVICE)
critic_for_pursuit = Critic(game="Hunt").to(device=DEVICE)
actor_fp_optim     = torch.optim.Adam(params=  actor_for_pursuit.parameters(), lr=LR_ACTOR)
critic_fp_optim    = torch.optim.Adam(params= critic_for_pursuit.parameters(), lr=LR_CRITIC)

actor_for_pursuit.train()
critic_for_pursuit.train()

logs_pursuit = single_agent_hunt_train_loop(env_pursuit, actor_for_pursuit, critic_for_pursuit, actor_fp_optim, critic_fp_optim)

plot_logs(logs_pursuit, window=100, file_name="pursuit_training_logs.png")

actor_for_pursuit.eval()

eval_logs_pursuit = single_agent_hunt_eval_loop(env_pursuit, actor_for_pursuit)

plot_logs(eval_logs_pursuit, file_name="pursuit_test_logs.png")

import pickle

print("Performance dual agent:")
print(f"AVG TOTAL REWARD:     {np.mean(eval_logs['total_reward_cumul']):>3.2f}")
print(f"AVG STAGS CAUGHT:     {np.mean(eval_logs['stag_catches']):>3.2f}")
print(f"AVG PLANTS HARVESTED: {np.mean(eval_logs['plants_harvested_cumul'])/2.0:>3.2f}")

print("Performance with random agent:")
print(f"AVG AGENT1 TOTAL REWARD: {np.mean(eval_logs_random['total_reward_1']):>3.2f}")
print(f"AVG STAGS CAUGHT:        {np.mean(eval_logs_random['stag_catches']):>3.2f}")
print(f"AVG PLANTS HARVESTED:    {np.mean(eval_logs_random['plants_harvested_1']):>3.2f}")

print("Performance with pursuit agent:")
print(f"AVG AGENT1 TOTAL REWARD: {np.mean(eval_logs_pursuit['total_reward_1']):>3.2f}")
print(f"AVG STAGS CAUGHT:        {np.mean(eval_logs_pursuit['stag_catches']):>3.2f}")
print(f"AVG PLANTS HARVESTED:    {np.mean(eval_logs_pursuit['plants_harvested_1']):>3.2f}")
