# %%
import os
import torch
import matplotlib.pyplot as plt
from env_wrapper import BoardsWrapper
from env_internals import BoardsImplementation
from agent_architecture import AgentParams, PPOAgent, RandomAgent, save_agents, load_agents
from misc_utils import smooth_list, find_latest_version
from training_loop import train_agents

# %%
size = 4
n_landmarks = 2
n_clues = 2
n_questions = 0
max_moves = size ** 2 * 4
history_len = 4
instant_reward_multiplier = 2.0
end_reward_multiplier = 10.0

env_seed = 12
sender_seed = 135
receiver_seed = 246
torch.manual_seed(45954)
torch.cuda.manual_seed(45954)

hidden_size = size ** 2 * 32
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

env_internals = BoardsImplementation(size, n_landmarks, n_clues, n_questions, seed = env_seed)
env = BoardsWrapper(env_internals, max_moves, history_len, instant_reward_multiplier, end_reward_multiplier, device)

sender_parameters = AgentParams(gamma = 0.99, alpha = 1e-4, gae_lambda = 0.95, policy_clip = 0.1, batch_size = 4, n_epochs = 5, seed = sender_seed)
receiver_parameters = AgentParams(gamma = 0.99, alpha = 1e-4, gae_lambda = 0.95, policy_clip = 0.1, batch_size = 4, n_epochs = 5, seed = receiver_seed)
sender_agent = PPOAgent(size, history_len, env.sender_n_actions, hidden_size, device, sender_parameters, input_channels=6)
receiver_agent = PPOAgent(size, history_len, env.receiver_n_actions, hidden_size, device, receiver_parameters)

# %%
model_folder = "./models"
os.makedirs(model_folder, exist_ok = True)
series_name = f"mixed-scheme_game-reflection_{size}x{size}_{n_landmarks}_{n_clues}_{n_questions}_{max_moves}_{history_len}"
agents_file_name = f"agents_{series_name}"

# %%
n_epochs = 10
n_episodes = 10_000
all_performances = list()
model_version = 0

for i in range(n_epochs):
    performances = train_agents(env, sender_agent, receiver_agent, n_episodes)
    smoothed_performances = smooth_list(performances, n_episodes // 20)
    all_performances.extend(performances)
    env.render()
    model_version += 1
    
    plt.figure(figsize = (12, 6))
    plt.scatter(range(n_episodes), performances)
    plt.xlabel("Episode")
    plt.ylabel("Performance")
    plt.title(f"Agent Performance over Episodes - Part {model_version}")
    plt.show()
    
    plt.figure(figsize = (12, 6))
    plt.plot(smoothed_performances)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Performance")
    plt.title(f"Agent Performance over Episodes with Smoothing - Part {model_version}")
    plt.show()
    
    save_agents(sender_agent, receiver_agent, os.path.join(model_folder, f"{agents_file_name}_iteration_{model_version}.pkl"))

smoothed_total = smooth_list(all_performances, len(all_performances) // 20)
plt.figure(figsize = (12, 6))
plt.plot(smoothed_total)
plt.xlabel("Episode")
plt.ylabel("Smoothed Performance")
plt.title(f"Agent Performance over Episodes with Smoothing")
plt.show()

sender_agent = PPOAgent(size, history_len, env.sender_n_actions, hidden_size, device, sender_parameters)
receiver_agent.freeze(True)

for i in range(n_epochs):
    performances = train_agents(env, sender_agent, receiver_agent, n_episodes)
    smoothed_performances = smooth_list(performances, n_episodes // 20)
    all_performances.extend(performances)
    env.render()
    model_version += 1
    
    plt.figure(figsize = (12, 6))
    plt.scatter(range(n_episodes), performances)
    plt.xlabel("Episode")
    plt.ylabel("Performance")
    plt.title(f"Agent Performance over Episodes - Part {model_version}")
    plt.show()
    
    plt.figure(figsize = (12, 6))
    plt.plot(smoothed_performances)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Performance")
    plt.title(f"Agent Performance over Episodes with Smoothing - Part {model_version}")
    plt.show()
    
    save_agents(sender_agent, receiver_agent, os.path.join(model_folder, f"{agents_file_name}_iteration_{model_version}.pkl"))

smoothed_total = smooth_list(all_performances, len(all_performances) // 20)
plt.figure(figsize = (12, 6))
plt.plot(smoothed_total)
plt.xlabel("Episode")
plt.ylabel("Smoothed Performance")
plt.title(f"Agent Performance over Episodes with Smoothing")
plt.show()

receiver_agent.freeze(False)

for i in range(n_epochs):
    performances = train_agents(env, sender_agent, receiver_agent, n_episodes)
    smoothed_performances = smooth_list(performances, n_episodes // 20)
    all_performances.extend(performances)
    env.render()
    model_version += 1
    
    plt.figure(figsize = (12, 6))
    plt.scatter(range(n_episodes), performances)
    plt.xlabel("Episode")
    plt.ylabel("Performance")
    plt.title(f"Agent Performance over Episodes - Part {model_version}")
    plt.show()
    
    plt.figure(figsize = (12, 6))
    plt.plot(smoothed_performances)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Performance")
    plt.title(f"Agent Performance over Episodes with Smoothing - Part {model_version}")
    plt.show()
    
    save_agents(sender_agent, receiver_agent, os.path.join(model_folder, f"{agents_file_name}_iteration_{model_version}.pkl"))

smoothed_total = smooth_list(all_performances, len(all_performances) // 20)
plt.figure(figsize = (12, 6))
plt.plot(smoothed_total)
plt.xlabel("Episode")
plt.ylabel("Smoothed Performance")
plt.title(f"Agent Performance over Episodes with Smoothing")
plt.show()

# %%
smoothed_total = smooth_list(all_performances, len(all_performances) // 600)
plt.figure(figsize = (12, 6))
plt.plot(smoothed_total)
plt.xlabel("Episode")
plt.ylabel("Smoothed Performance")
plt.title(f"Agent Performance over Episodes with Smoothing")
plt.show()

# %%



