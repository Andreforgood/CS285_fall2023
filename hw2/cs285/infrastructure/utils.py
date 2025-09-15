from collections import OrderedDict
import numpy as np
import copy
from cs285.networks.policies import MLPPolicy
import gym
import cv2
from cs285.infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List

############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]: # -> means the function returns a dictionary with string keys and numpy array values
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()        # reset the environment to start a new episode
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], [] # lists to store the trajectory data
    steps = 0               # counter for the number of steps taken in the trajectory
    while True:
        # render an image  可视化环境，把 Gym 里的仿真画面渲染成图像（RGB array）
        if render: 
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1] 
                # render 的作用是 从指定相机视角拍一张 RGB 图像。
                # arr[::-1] → 把整个数组倒序。
	            # 如果是 2D/3D 数组（比如图像 (H, W, 3)），默认作用在第一维（行）。
            else:
                img = env.render(mode="single_rgb_array") # mode 指定 render 的输出格式
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) # resize to 250x250, interpolation means the method of resizing.
            )

        # TODO use the most recent ob and the policy to decide what to do
        ac: np.ndarray = policy.get_action(ob) # shape (ac_dim,)

        # TODO: use that action to take a step in the environment
        next_ob, rew, done, _ = env.step(ac) # take a step in the environment using the action ac
        # next_ob: np.ndarray, shape (ob_dim,)
        # rew: float, the reward for taking that action
        # done: bool, whether the episode has ended
        # _ : dict, diagnostic information from the environment, not used here
        
        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done: bool = done or (steps >= max_length)

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]: # returns a tuple: (list of trajectory dictionaries, total timesteps collected)
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj) # append the collected trajectory dictionary to the list of trajectories

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs] # list of total rewards for each training trajectory
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs] # list of total rewards for each evaluation trajectory
    # eval_trajs: list of trajectory dictionaries collected during evaluation

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs] # list of lengths (in timesteps) for each training trajectory
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict() # OrderedDict is like a regular dict, but maintains the order of insertion
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])              # shape (sum of all traj lengths, ob_dim)
    actions = np.concatenate([traj["action"] for traj in trajs])                        # shape (sum of all traj lengths, ac_dim)
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])    # shape (sum of all traj lengths, ob_dim)
    terminals = np.concatenate([traj["terminal"] for traj in trajs])                    # shape (sum of all traj lengths,)
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])           # shape (sum of all traj lengths,)
    unconcatenated_rewards = [traj["reward"] for traj in trajs]                # list of arrays, each array is the rewards for a trajectory
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])
