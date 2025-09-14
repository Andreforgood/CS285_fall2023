from typing import Optional, Sequence # Sequence is a generic version of list/tuple
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,                                # dimension of observation space
        ac_dim: int,                                # dimension of action space
        discrete: bool,                             # whether the action space is discrete or continuous
        n_layers: int,                              # number of layers in the policy network
        layer_size: int,                            # size of each layer in the policy network
        gamma: float,                               # discount factor
        learning_rate: float,                       # learning rate for the policy network
        use_baseline: bool,                         # whether to use a baseline (value function) for advantage estimation
        use_reward_to_go: bool,                     # whether to use reward-to-go for Q estimation
        baseline_learning_rate: Optional[float],    # learning rate for the baseline network
        baseline_gradient_steps: Optional[int],     # number of gradient steps for the baseline network
        gae_lambda: Optional[float],                # lambda for GAE
        normalize_advantages: bool,                 # whether to normalize advantages
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate 
        ) # define a PyTorch module neural network, a MLP

        # create the critic (baseline) network as a MLP
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps # number of gradient steps to take when updating the baseline
        else:
            self.critic = None 

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],          # list of arrays, each array is the observations for a trajectory
        actions: Sequence[np.ndarray],      # list of arrays, each array is the actions for a trajectory
        rewards: Sequence[np.ndarray],      # list of arrays, each array is the rewards for a trajectory
        terminals: Sequence[np.ndarray],    # list of arrays, each array is the terminals for a trajectory
    ) -> dict:                              # returns a dictionary of info about the update
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards) 
        # q_values: 后面跟 Sequence[np.ndarray] 表示：这个变量 q_values 是一个“numpy 数组的序列”

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs, axis=0)               # np.concatenate: 将多个数组沿指定轴连接起来，形成一个新的数组 (B, obs_dim)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages) # used update function in MLPPolicyPG we imported
        
        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline 
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = {}
            for _ in range(self.baseline_gradient_steps):
                self.critic.update(obs, q_values)
            info.update(critic_info)            # merge two dictionaries 其实是 Python 字典的方法调用，把 critic_info 里面的键值对合并到 info 里

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]: # helper function, used internally.
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = [
                np.array(self.discounted_return(rewards_i), dtype=np.float32)  # each for one trajectory
                for rewards_i in rewards
                ]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [
                np.array(self.discounted_reward_to_go(rewards_i), dtype=np.float32)
                for rewards_i in rewards
            ]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            values = self.critic(ptu.from_numpy(obs)).cpu().numpy().squeeze() # 自动调用 critic 的 forward 函数
            #整体流程是：
	        #1.	把 numpy 的 obs 转成 torch tensor → ptu.from_numpy(obs)
	        #2.	喂给 critic 网络，得到预测值 V(s) → self.critic(...)
	        #3.	转回 numpy → .cpu().numpy()
	        #4.	把 shape 从 (B,1) 压成 (B,) → .squeeze()

            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]  
                lam = self.gae_lambda

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])         # 这样在循环里访问 values[i+1]（对应 V_{t+1}）时，就不会越界。
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    nonterminal = 1 - terminals[i]
                    delta = rewards[i] + self.gamma * values[i + 1] * nonterminal - values[i]
                    advantages[i] = delta + self.gamma * lam * nonterminal * advantages[i + 1]

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        G = 0.0
        pow_gamma = 1.0
        for r in rewards:
            G += pow_gamma * r
            pow_gamma *= self.gamma
        discounted_returns = [G] * len(rewards)
        return discounted_returns


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        T = len(rewards)
        discounted_rewards = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            discounted_rewards[t] = running
        
        return discounted_rewards.tolist()
