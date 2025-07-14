import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.crab_dim = observation_space["crabs"].shape[0]
        self.months_dim = 12
        input_dim = self.crab_dim + self.months_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, features_dim), nn.ReLU()
        )

    def forward(self, obs):
        crabs = obs["crabs"]
        months = F.one_hot(obs["months"].long() - 1, num_classes=12).float()
        x = torch.cat([crabs, months], dim=1)
        return self.net(x)


class LipschitzPPO(PPO):
    def __init__(self, *args, gp_coef=1.0, gp_K=0.0, **kwargs):
        self.gp_coef = gp_coef
        self.gp_K = gp_K
        super().__init__(*args, **kwargs)

    def train(self):
        entropy_losses, all_values, all_log_prob, clip_fractions = [], [], [], []
        clip_range = self.clip_range(self._current_progress_remaining)
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Box):
                    actions = torch.clamp(actions, torch.as_tensor(self.action_space.low).to(actions.device), torch.as_tensor(self.action_space.high).to(actions.device))

                obs = rollout_data.observations
                obs_gp = {
                    "crabs": obs["crabs"].detach().clone().requires_grad_(True),
                    "months": obs["months"]
                }

                dist = self.policy.get_distribution(obs_gp)
                mean_action = dist.distribution.mean
                gp_loss = 0.0

                # for j in range(mean_action.shape[1]):
                #     grad = torch.autograd.grad(mean_action[:, j].sum(), obs_gp["crabs"], create_graph=True)[0]
                #     grad_norm = (grad ** 2).sum(dim=1).sqrt()
                #     excess = (grad_norm - self.gp_K).clamp(min=0.0)
                #     gp_loss += (excess ** 2).mean()
                #     # gp_loss += (grad ** 2).sum(dim=1).mean() default to gp_K = 0
                # gp_loss = gp_loss / mean_action.shape[1]
                try:
                    for j in range(mean_action.shape[1]):
                        grad = torch.autograd.grad(mean_action[:, j].sum(), obs_gp["crabs"], create_graph=True)[0]
                        if not torch.isfinite(grad).all():
                            raise RuntimeError("NaN in gradient of mean_action w.r.t. obs")
                
                        grad_norm = grad.norm(dim=1)
                        excess = (grad_norm - self.gp_K).clamp(min=0.0)
                        gp_loss += (excess ** 2).mean()
                    gp_loss = gp_loss / mean_action.shape[1]
                except Exception as e:
                    print(f"GP loss error: {e}")
                    gp_loss = torch.tensor(0.0, device=obs["crabs"].device)


                values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(rollout_data.returns, values)
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.gp_coef * gp_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                entropy_losses.append(entropy_loss.item())
                all_values.append(values.mean().item())
                all_log_prob.append(log_prob.mean().item())
                clip_fractions.append(((ratio - 1.0).abs() > clip_range).float().mean().item())

        self._n_updates += self.n_epochs
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_gradient_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/gp_loss", gp_loss.item())
        self.logger.record("train/approx_kl", (rollout_data.old_log_prob - log_prob).mean().item())
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))


