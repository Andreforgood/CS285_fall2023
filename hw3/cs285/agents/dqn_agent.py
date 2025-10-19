from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int], # e.g. (4,) for CartPole-v1
        num_actions: int, # e.g. 2 for CartPole-v1
        make_critic: Callable[[Tuple[int, ...], int], nn.Module], # 输入 obs shape 和 action 数量，返回一个 Q 网络
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ], 
        discount: float,
        target_update_period: int,
        use_double_q: bool = False, 
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        if np.random.rand() < epsilon:
            action = torch.tensor([np.random.randint(self.num_actions)], device=ptu.device, dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = self.critic(observation) # Shape (1, num_actions)
                action = torch.argmax(q_values, dim=1) # Shape (1,)
        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            next_qa_values_tgt = self.target_critic(next_obs) # Shape (batch_size, num_actions)

            if self.use_double_q:
                # ------- Double DQN 核心 -------
                # 1) 在线网络在 s' 上选动作（只用于选择，不用于打分）
                next_qa_online = self.critic(next_obs)                       # (B, A)
                next_action = torch.argmax(next_qa_online, dim=1)            # (B,)

                 # 2) 目标网络对“在线网络选出来的动作”打分
                next_q_values = next_qa_values_tgt.gather(1, next_action.unsqueeze(1)).squeeze(1)  # (B,)

            else:
                next_action = torch.argmax(next_qa_values_tgt, dim=1) # Shape (batch_size,)
                # 取出下一时刻的最大 Q 值
                next_q_values = next_qa_values_tgt.gather(1, next_action.unsqueeze(1)).squeeze(1) # Shape (batch_size,)
            # # Bellman target: r + γ * (1-done) * max_a' Q_target(s', a')
            target_values = reward + self.discount * (1 - done) * next_q_values

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs) # Shape (batch_size, num_actions)
        q_values = qa_values.gather(1, action.unsqueeze(1)).squeeze(1) # Shape (batch_size,)
        # Compute from the data actions; see torch.gather
        loss = self.critic_loss(q_values, target_values)


        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()
            
        return critic_stats
