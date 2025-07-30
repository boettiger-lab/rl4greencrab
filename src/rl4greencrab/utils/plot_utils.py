import numpy as np
import pandas as pd
import os
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from rl4greencrab.utils.simulate import *

def environment_simulation(env, agent, 
                           reps=10, 
                           obs_names = None, 
                           acts_names = None, 
                           save_df=False, 
                           save_path='.', 
                           agent_name = 'ppo'):
    num_obs = np.prod(len(env.observation_space))
    num_acts = np.prod(env.action_space.shape)
    obs_names = obs_names or [f'obs{i}' for i in range(num_obs)]
    acts_names = acts_names or [f'act{i}' for i in range(num_acts)]
    #
    data = {
        't': [],
        **{obsn: [] for obsn in obs_names},
        **{actn: [] for actn in acts_names},
        'rew': [],
        'rep': [],
        'crab_pop':[]
    }
    env = env
    agent = agent
    for rep in range(reps): # try score as average of 100 replicates, still a noisy measure
        episode_reward = 0.0
        observation, _ = env.reset()
        for t in range(env.Tmax):
            action, _ = agent.predict(observation, deterministic=True)
            #
            data['rew'].append(episode_reward)
            data['rep'].append(rep)
            data['t'].append(t)
            data['crab_pop'].append(env.state)
            for idx, obs_name in enumerate(obs_names):
                data[obs_name].append(observation['crabs'][idx])
            for idx, act_name in enumerate(acts_names):
                data[act_name].append(action[idx])
            #
            observation, reward, terminated, done, info = env.step(action)
            episode_reward += reward
            #
            if terminated or done:
                break
    if save_df:
        df = pd.DataFrame(data)
        DATAPATH = save_path
        df.to_csv(os.path.join(DATAPATH,f"{agent_name}_env.csv"), index = False)
    return data

# plot for change of crab population of certain size overtime
def plot_selected_sizes(expanded_df:pd.DataFrame, selected_sizes, 
                        title = "Green Crab Population Change Over Time", 
                        xlabel= "Time (t)", 
                        ylabel= "Population", 
                        legend_title = "Crab Sizes"):
    plt.figure(figsize=(12, 8))
    time = expanded_df['t']  # Time column

    # If no sizes selected, show a placeholder message
    if not selected_sizes:
        plt.text(0.5, 0.5, 'No sizes selected', fontsize=20, ha='center', va='center')
        plt.axis('off')
        plt.show()
        return

    # Plot each selected size
    for col in selected_sizes:
        plt.plot(time, expanded_df[col], label=col)

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()

    # Show the plot
    plt.show()


# plots for policy agents
def agent_action_plot(evalEnv, agent, action=[], name='default'):
    data = simulator(env=evalEnv, agent=agent).simulate_full_named_dict_obs_acts()
    df = pd.DataFrame(data)
    
    # Line plots (separate)
    df[df.rep == 0].plot(x='t', y=['act0', 'act1', 'act2'], title="Actions over Time")
    df[df.rep == 0].plot(x='t', y=['obs0'], title="Observation 0 over Time")
    df[df.rep == 0].plot(x='t', y=['obs1'], title="Observation 1 over Time")

    # Combined scatter plot (in one figure)
    fig, ax = plt.subplots()
    ax.scatter(x=df[df.rep == 0]['obs0'], y=df[df.rep == 0]['act0'], label='act0 vs obs0', alpha=0.7)
    ax.scatter(x=df[df.rep == 0]['obs0'], y=df[df.rep == 0]['act1'], label='act1 vs obs0', alpha=0.7)
    ax.scatter(x=df[df.rep == 0]['obs0'], y=df[df.rep == 0]['act2'], label='act2 vs obs0', alpha=0.7)
    ax.set_xlabel('Observation 0')
    ax.set_ylabel('Action')
    ax.set_title('Scatter Plots: act0/act1 vs obs0')
    ax.legend()
    
    plt.show()
    return df

# Gaussian Smoothing Function

#@ray.remote
def GaussianProcessPolicy(policy_df, length_scale=1, noise_level=0.1):
  """
  policy_df.columns = [X, Y, Z, act_x, act_y]
                    -> action (act_x, act_y) taken at point (X, Y, Z)
  """
  predictors = policy_df[["obs0", "obs1"]].to_numpy()
  targets = policy_df[["act0", "act1", 'act2']].to_numpy()
  kernel = (
    1.0 * RBF(length_scale = length_scale) 
    + WhiteKernel(noise_level=noise_level)
    )
  print("Fitting Gaussian Process...")
  gpp = (
    GaussianProcessRegressor(kernel=kernel, random_state=0)
    .fit(predictors, targets)
    )
  print("Done fitting Gaussian Process...")
  return gpp
    
# generate gaussian smoothed action and observations df
def generate_gpp_episodes(gpp, env, reps=50):
      """ gpp is a gaussian process regression of the RL policy """
      df_list = []
      state_list = []
      for rep in range(reps):
        episode_reward = 0
        observation, _ = env.reset()
        population = env.observation['crabs']
        state = env.state
        for t in range(env.Tmax):
            
          action = gpp.predict([population])[0]

          esc_x = population[0] * (1 - action[0])
          esc_y = population[1] * (1 - action[1])
          
          df_list.append(np.append([t, rep, action[0], action[1], action[2], episode_reward, esc_x], population))
          state_list.append([t, rep, state])
        
          observation, reward, terminated, done, info = env.step(action)
          population = env.observation['crabs']
          state = env.state
          
          episode_reward += reward
          if terminated:
            break
      cols =['t','rep', 'act0', 'act1', 'act2', 'rew', 'esc_x', 'obs0', 'obs1']
      df = pd.DataFrame(df_list, columns = cols)
      state_df = pd.DataFrame(state_list, columns= ['t', 'rep', 'state'])
      df = pd.merge(df, state_df, on=['t','rep'], how='inner')
      return df, state_df

# generate rews distrbution of different agent
""" dict exmaple: agent_dict = {
    'td3Agent': td3Agent,
    'ppoAgent': ppoAgent,
    'tqcAgent': tqcAgent,
    'recurrentPPOAgent': recurrentPPOAgent,
    'lppoAgent': lppoAgent,
    'constantAgent': multiConstAction(env=env, action=np.array([83.87232800633504, 596.3225575635984, 14.882297944474463]))
}"""
# cols = [t, rep, agent1_rew ...]
def generate_policy_rews_df_agents(agent_dict, eval_env, rep=3):
    dfs = []
    for name, agent in agent_dict.items():
        df = pd.DataFrame(environment_simulation(eval_env, agent, reps=3))
        df = df[df['rep']<3].loc[:, ['t', 'rep', 'rew']]
        tmp = (
            df
            .rename(columns={'rew': f"{name}_rew"})
            .set_index(['t','rep'])
        )
        dfs.append(tmp)
    
    # 2) Concatenate them side‑by‑side, then bring t, rep back as columns
    rews_df = pd.concat(dfs, axis=1).reset_index()
    return rews_df

# generate rews distrbution of different agent
# cols = [t, rep, agent1_rew ...]
def generate_policy_rews_df_dfs(df_dicts, rep=3, save_dir='.', df_name='rews_df'):
    dfs = []
    for name, df in df_dicts.items():
        df = df[df['rep']<3].loc[:, ['t', 'rep', 'rew']]
        tmp = (
            df
            .rename(columns={'rew': f"{name}_rew"})
            .set_index(['t','rep'])
        )
        dfs.append(tmp)
    
    # 2) Concatenate them side‑by‑side, then bring t, rep back as columns
    rews_df = pd.concat(dfs, axis=1).reset_index()
    rews_df.to_csv(os.path.join(save_dir,f"{df_name}.csv"), index = False)
    return rews_df

# comparing reward distribution between rl_agents and constant agent
def agent_rew_vs_constant_rew_plot(model_rews_dict, constant_rews, save_dir='.'):
    colors = {
        'PPO': 'blue',
        'TD3': 'orange',
        'TQC': 'green',
        'RecurrentPPO': 'purple',
        'LipschitzPPO': 'yellow',
        'Constant': 'red'
    }
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (model_name, rewards) in enumerate(models.items()):
        ax = axes[i]
        # Plot current model
        ax.hist(rewards, alpha=0.6, color=colors[model_name], label=model_name)
        ax.axvline(np.mean(rewards), color=colors[model_name], linestyle='dashed', linewidth=2, label=f'{model_name} Mean')
    
        # Plot constant model
        ax.hist(constant_rewards, alpha=0.3, color=colors['Constant'], label='Constant')
        ax.axvline(np.mean(constant_rewards), color=colors['Constant'], linestyle='dashed', linewidth=2, label='Constant Mean')
    
        ax.set_title(f"{model_name} vs Constant")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

    # 3. Save to disk
    save_path = os.path.join(save_dir, 'rews_comparision.png')
    fig.savefig(save_path,      # filename (relative or absolute path)
                dpi=300,            # resolution in dots per inch
                bbox_inches='tight',# trim whitespace
                transparent=False)  # make background transparent
    plt.tight_layout()
    plt.show()

# plot heatmap relative to crab size and time
def state_heatmap(df, rep=0, use_log=True):
    state_rep = df[df['rep']==rep]
    state_rep = state_rep.loc[:, ['t','crab_pop']]
    state_rep = state_rep.set_index('t')
    state_rep['t'] = state_rep.index
    if use_log:
        if isinstance(state_rep['crab_pop'][0], str):
            state_rep['state'] = state_rep.loc[:, 'crab_pop'].apply(
                lambda pop_list: np.log(np.array(np.fromstring(pop_list.strip("[]"), sep=' '), dtype=np.float64)+ 1)
            )
        else:
            state_rep['state'] = state_rep.loc[:, 'crab_pop'].apply(
                lambda pop_list: np.log(np.array(pop_list, dtype=np.float64)+ 1)
            )
    else:
        state_rep['state'] =  state_rep.loc[:, 'crab_pop'].apply(
                lambda pop_list:np.array(np.fromstring(pop_list.strip("[]"), sep=' '), dtype=np.float64)
            )
    state_rep = state_rep.loc[:, ['t','state']]
    size_df = pd.DataFrame(state_rep['state'].tolist())
    size_df.columns = [f'size_{i}' for i in range(1, size_df.shape[1]+1)]
    state_rep =  pd.concat([state_rep.drop(columns=['state']), size_df], axis=1)
    # prepare heatmap
    df_melted = state_rep.melt(id_vars='t', value_name='abundance', var_name='size')
    df_melted['size'] = df_melted['size'].str.extract(r'(\d+)').astype(int)
    heatmap_data = df_melted.pivot(index='size', columns='t', values='abundance')
    # display(heatmap_data)
    fig = plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Abundance'})
    plt.xlabel('Time')
    plt.ylabel('Size Class')
    plt.title('Heatmap of State Abundance Over Time by Size')
    plt.savefig("my_seaborn_plot.png")
    plt.show()
    return fig