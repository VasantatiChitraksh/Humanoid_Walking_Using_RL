import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        self.actor_mu = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, act_dim),
            nn.Tanh()
        )

        # revert to old name
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

        # revert to old critic name
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).expand_as(mu)
        return mu, std

    def value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)

        if action is None:
            action = dist.rsample()

        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().mean(-1)
        v = self.value(x)

        return action, logp, entropy, v
