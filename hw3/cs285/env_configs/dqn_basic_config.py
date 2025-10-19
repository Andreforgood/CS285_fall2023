from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

from cs285.env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
import cs285.infrastructure.pytorch_util as ptu

def basic_dqn_config( # 输入一堆超参数（有默认值），返回一个 dict，里面放了：
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    total_steps: int = 300000,
    discount: float = 0.99,
    target_update_period: int = 1000,
    clip_grad_norm: Optional[float] = None,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 128,
    **kwargs
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module: 
        # 输入 obs shape 和 action 数量，返回一个 Q 网络
        # 这里的 observation_shape 是 env.observation_space.shape，比如 (4,) for CartPole-v1
        return ptu.build_mlp(
            input_size=np.prod(observation_shape),
            output_size=num_actions,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    # ConstantLR 是 PyTorch 的学习率调度器之一。
	# factor=1.0 表示恒定学习率（不变）。等价于“先留一个钩子”，将来可换成 Warmup/StepLR/Cosine 等，不需要改训练循环。

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1),
            (total_steps * 0.1, 0.02),
        ],
        outside_value=0.02,
    )
    # PiecewiseSchedule（你们项目自带）表示分段线性插值的标量函数：步数 t=0 时 ε=1（疯狂探索）
	# 到 t=total_steps*0.1 线性降到 0.02
	# 超出该范围就固定为 outside_value=0.02

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(gym.make(env_name, render_mode="rgb_array" if render else None))

    # gym.make(...)：创建环境。
	# render_mode="rgb_array"：需要渲染时返回帧数组（适合离线保存/视频，不会弹窗）。
	# RecordEpisodeStatistics：Gym 的 wrapper，会把每局（episode）的统计（长度、回报等）记录到 env.return_queue / length_queue，
    # 同时 env.step(...) 的 info 里也会带上 episode 信息，方便 logger 读。

    log_string = "{}_{}_s{}_l{}_d{}".format(
        exp_name or "dqn",
        env_name,
        hidden_size,
        num_layers,
        discount,
    )

    if use_double_q:
        log_string += "_doubleq"

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "exploration_schedule": exploration_schedule,
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }
    
    
    
# what is Q network?
# Q network is a neural network that approximates the Q-function in reinforcement learning.
# It's used when the state and action spaces are large or continuous, making it impractical to use a Q-table.