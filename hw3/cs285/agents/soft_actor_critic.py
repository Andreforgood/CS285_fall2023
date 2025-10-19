from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        # TODO(student): Implement the different backup strategies.
        if self.target_critic_backup_type == "doubleq":
            # 典型 Double-Q：每个评论家用“另一个”的估计做目标（2个评论家时最常见）
            # 泛化做法：将 next_qs 循环右移一位作为“对方”的估计
            if self.num_critic_networks < 2:
                # 只有一个评论家就退化为 mean（或直接返回）
                next_qs = next_qs
            else:
                perm = list(range(self.num_critic_networks))
                perm = perm[-1:] + perm[:-1]  # 右移一位
                next_qs = next_qs[perm, :]  # (num_critics, B)

        elif self.target_critic_backup_type == "min":
            # Clipped Double Q：对所有 target-critics 取最小，避免过估计
            # 结果是 (B,)；下面会 broadcast 回 (num_critics, B)
            next_qs = torch.min(next_qs, dim=0)[0]  # (B,)

        elif self.target_critic_backup_type == "mean":
            # 多评论家取均值作为目标，更平滑稳健（REDQ 也常这么做）
            next_qs = torch.mean(next_qs, dim=0)    # (B,)

        else:
            # "redq" 或其它策略你也可以在这里拓展；
            # 默认直接返回每个评论家的各自估计
            pass

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # TODO(student)
            # Sample from the actor
            # 目标分支无需梯度，用 sample() 或 rsample() 都可以
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
            next_action = next_action_distribution.sample() # (B, action_dim)

            # Compute the next Q-values for the sampled actions
            next_qs = self.target_critic(next_obs, next_action) # (num_critics, B)
            # 用 target critic(s) 评估 Q_target(s', a')

            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            next_qs = self.q_backup_strategy(next_qs)
            
            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(student): Add entropy bonus to the target values for SAC
                next_action_entropy = self.entropy(next_action_distribution)  # (B,)
                next_qs += next_action_entropy[None, :] * self.temperature

            # Compute the target Q-value
            # Bellman 目标：y = r + γ(1-d) * next_q
            not_done = 1.0 - done.float()
            target_values: torch.Tensor = reward[None, :] + self.discount * not_done[None, :] * next_qs
            
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size
            )

        # TODO(student): Update the critic
        # Predict Q-values
        q_values = self.critic(obs, action)  # (num_critics, B)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """

        # TODO(student): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        try:
            ent = action_distribution.entropy()
            # 某些分布可能返回 (B, action_dim)；对维度求和得到每个样本的总熵
            if ent.ndim > 1:
                ent = ent.sum(-1)
            return ent
        except Exception:
            # 回退到蒙特卡洛估计：H = -E[log π(a|s)]
            K = max(1, self.num_actor_samples)
            # 用 rsample 保留梯度到分布参数（对 actor 损失很重要）
            samples = action_distribution.rsample((K,))             # (K, B, A)
            logp = action_distribution.log_prob(samples)            # (K, B)（假设 Independent 包装）
            if logp.ndim > 2:  # 以防返回 (K,B,action_dim)
                logp = logp.sum(-1)
            ent = -logp.mean(dim=0)                                 # (B,)
            return ent

    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # TODO(student): Generate an action distribution
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        with torch.no_grad():
            # TODO(student): draw num_actor_samples samples from the action distribution for each batch element
            K = max(1, self.num_actor_samples)
            action = action_distribution.sample((K,))  # (K, B, A)
            assert action.shape == (
                self.num_actor_samples,
                batch_size,
                self.action_dim,
            ), action.shape

            # TODO(student): Compute Q-values for the current state-action pair
            # 计算 Q(s, a_k)，注意把 (K,B,·) 展平喂入 critic，再 reshape 回来
            obs_expand = obs.unsqueeze(0).expand(K, *obs.shape).contiguous().view(K * batch_size, *obs.shape[1:])
            act_flat  = action.contiguous().view(K * batch_size, self.action_dim)

            qs = self.critic(obs_expand, act_flat)  # (num_critics, K*B)
            qs = qs.view(self.num_critic_networks, K, batch_size)  # (C, K, B)
            
            assert qs.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(qs, axis=0)
            advantage = q_values

        # Do REINFORCE: calculate log-probs and use the Q-values
        # TODO(student)
        # 计算 log π(a_k|s)（带梯度）
        log_probs = action_distribution.log_prob(action)  # (K, B)
        if log_probs.ndim > 2:  # 以防返回 (K,B,action_dim)
            log_probs = log_probs.sum(-1)

        # REINFORCE 损失： -E[ A * log π ]，A 不反传到 critic
        loss = -(advantage.detach() * log_probs).mean()

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # TODO(student): Sample actions
        
        # 1) 分布 & 进行 reparameterized 采样（允许路径导数）
        action_distribution: torch.distributions.Distribution = self.actor(obs)
        action = action_distribution.rsample()  # (B, A)

        # 2) 计算 Q(s, a)，对评论家取均值
        qs = self.critic(obs, action)          # (num_critics, B)
        q_mean = qs.mean(dim=0)                # (B,)

        # 3) reparam 版本的 actor loss：最大化 Q ≡ 最小化 -Q
        #    熵项（α * H）由 update_actor 外层统一加入
        loss = -q_mean.mean()

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []
        # TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
        for _ in range(self.num_critic_updates):
            critic_info = self.update_critic(
                obs=observations,
                action=actions,
                reward=rewards,
                next_obs=next_observations,
                done=dones,
            )
            critic_infos.append(critic_info)
            
        # TODO(student): Update the actor
        actor_info = self.update_actor(obs=observations)

        # TODO(student): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        if self.soft_target_update_rate is not None:
            # 软更新：每步用 tau 做指数滑动
            self.soft_update_target_critic(self.soft_target_update_rate)
        elif self.target_update_period is not None:
            # 硬更新：每隔 N 步同步一次
            if step % self.target_update_period == 0:
                self.update_target_critic()
                
        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }
