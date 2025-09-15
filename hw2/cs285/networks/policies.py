import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,        
        learning_rate: float,
    ):
        super().__init__()

        if discrete:                                    # 用 logits_net 输出每个动作的 logits（未归一化对数概率），后面用 Categorical 分布。
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:                                           # 用 mean_net 输出均值，用一个独立的参数向量 logstd 表示对数标准差，后面用 Normal 分布。
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        if isinstance(obs, np.ndarray) and obs.ndim == 1:
            obs = obs[None, :]                              # add a batch dimension (1, ob_dim)
        obs_t = ptu.from_numpy(obs)                         # 返回的 obs_t 是 torch.FloatTensor，shape (B, ob_dim)。
        dist = self.forward(obs_t)                           # forward 返回一个 分布对象（torch.distributions.Distribution），
        if self.discrete:
            # Categorical: sample shape (B, ac_dim)
            action = dist.sample()
            action_np = action.cpu().numpy()                # shape (B,)
            
            return action_np[0] if action_np.shape[0] == 1 else action_np
        else:
            # Normal: sample shape (B, ac_dim)
            action = dist.sample()
            action_np = action.cpu().numpy()
            
            return action_np[0] if action_np.shape[0] == 1 else action_np
            

    def forward(self, obs: torch.FloatTensor): # obs: shape (B, ob_dim)
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)                 # shape (B, ac_dim)   每个entry对应一个动作的 logits（分布）
            dist = distributions.Categorical(logits=logits) # shape (B, ac_dim)  每个 entry 对应一个动作的概率分布
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)                    # shape (B, ac_dim)
            std = torch.exp(self.logstd)                 # shape (ac_dim,) 不随 observation 变化
            dist = distributions.Normal(mean, std)       # 假设每个动作维度独立同分布，with broadcast
        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError # 基类不实现，强制子类实现


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        dist = self.forward(obs)
        if self.discrete:
            log_probs = dist.log_prob(actions)  # shape (B,)
        
        else:
            log_probs = dist.log_prob(actions).sum(axis=-1) 
            # 各维独立 ⇒ 整个动作向量的 log 概率是逐维相加：dist.log_prob(actions) 会给出形状 (B, ac_dim) 的逐维 log-prob，需要 .sum(dim=-1) 合成 (B,)
            
        loss = -(log_probs * advantages).mean()     # 负号是因为要 maximize    
            
        self.optimizer.zero_grad()                  # 清空上一次迭代的梯度缓存
        loss.backward()                             # 反向传播：根据当前的 loss 计算每个参数的梯度
        self.optimizer.step()                       # 用计算出的梯度更新参数

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }






"""
在离散环境(比如 CartPole)中,动作只有有限个整数编号,比如 {0,1}。

import torch
from torch.distributions import Categorical

# 假设 policy 的输出是动作概率 [0.2, 0.8]
probs = torch.tensor([0.2, 0.8])
dist = Categorical(probs=probs)

# 抽一个样本
print(dist.sample())   # tensor(1)，因为 1 的概率 0.8 较高

每次 dist.sample() 返回一个 整数动作编号。
如果输入是 batch:

probs = torch.tensor([[0.2, 0.8],
                      [0.6, 0.4]])
dist = Categorical(probs=probs)

print(dist.sample())   # tensor([1, 0])  → 两个状态分别采的动作
 Shape: (B,)
B = batch size
每个位置就是一个动作编号(int)。
"""


"""
基类 MLPPolicy 只负责定义通用的网络结构(forward、get_action 等），但是不定义怎么训练它。

因为：
如果是 PG(Policy Gradient) 算法 → update 就是 maximize advantage * logπ
如果是 Actor-Critic 算法 → update 就是 policy loss + critic loss
如果是 DQN → 其实不用 policy 直接做 value update

所以：不同 RL 算法的 policy 更新规则 完全不同。
基类 MLPPolicy 不写 update,强制子类(比如 MLPPolicyPG)自己实现。

这就是 面向对象设计里的 “模板模式”:基类只规定接口(update),不同子类实现不同细节。
"""