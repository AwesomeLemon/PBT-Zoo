import collections

import math
from typing import Any, Callable, Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F

class Agent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(self,
                 policy_layers: Sequence[int],
                 value_layers: Sequence[int],
                 entropy_cost: float,
                 discounting: float,
                 reward_scaling: float,
                 lambda_: float,
                 ppo_epsilon: float,
                 device: str):
        super(Agent, self).__init__()

        policy = []
        for w1, w2 in zip(policy_layers, policy_layers[1:]):
            policy.append(nn.Linear(w1, w2))
            policy.append(nn.SiLU())
        policy.pop()  # drop the final activation
        self.policy = nn.Sequential(*policy)

        value = []
        for w1, w2 in zip(value_layers, value_layers[1:]):
            value.append(nn.Linear(w1, w2))
            value.append(nn.SiLU())
        value.pop()  # drop the final activation
        self.value = nn.Sequential(*value)

        self.register_buffer('num_steps', torch.zeros((), device=device))
        self.register_buffer('running_mean', torch.zeros(policy_layers[0], device=device))
        self.register_buffer('running_variance', torch.zeros(policy_layers[0], device=device))

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = lambda_
        self.epsilon = ppo_epsilon
        self.device = device

    @torch.jit.export
    def dist_create(self, logits):
        """Normal followed by tanh.

        torch.distribution doesn't work with torch.jit, so we roll our own."""
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + .001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(self, loc, scale):
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x):
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc, scale):
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(self, loc, scale, dist):
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation):
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance = self.running_variance + var_diff

    @torch.jit.export
    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    @torch.jit.export
    def get_logits_action(self, observation):
        observation = self.normalize(observation)
        logits = self.policy(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    @torch.jit.export
    def compute_gae(self, truncation, termination, reward, values,
                    bootstrap_value):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        deltas = reward + self.discounting * (
                1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)

        for ti in range(truncation_mask.shape[0]):
            ti = truncation_mask.shape[0] - ti - 1
            acc = deltas[ti] + self.discounting * (
                    1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc
            vs_minus_v_xs[ti] = acc

        # Add V(x_s) to get v_s.
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
        advantages = (reward + self.discounting *
                      (1 - termination) * vs_t_plus_1 - values) * truncation_mask
        return vs, advantages

    @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor]):
        observation = self.normalize(td['observation'])
        policy_logits = self.policy(observation[:-1])
        baseline = self.value(observation)
        baseline = torch.squeeze(baseline, dim=-1)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = baseline[-1]
        baseline = baseline[:-1]
        reward = td['reward'] * self.reward_scaling
        termination = td['done'] * (1 - td['truncation'])

        loc, scale = self.dist_create(td['logits'])
        behaviour_action_log_probs = self.dist_log_prob(loc, scale, td['action'])
        loc, scale = self.dist_create(policy_logits)
        target_action_log_probs = self.dist_log_prob(loc, scale, td['action'])

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td['truncation'],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value)

        rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon,
                                     1 + self.epsilon) * advantages
        policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = torch.mean(self.dist_entropy(loc, scale))
        entropy_loss = self.entropy_cost * -entropy

        return policy_loss + v_loss + entropy_loss


StepData = collections.namedtuple(
    'StepData',
    ('observation', 'logits', 'action', 'reward', 'done', 'truncation'))


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        items[k] = f(*[sd._asdict()[k] for sd in sds])
    return StepData(**items)


def unroll_first(data):
    data = data.swapaxes(0, 1)
    return data.reshape([data.shape[0], -1] + list(data.shape[3:]))


def eval_unroll(agent, env, length):
    """Return number of episodes and average reward for a single unroll."""
    observation = env.reset()
    episodes = torch.zeros((), device=agent.device)
    episode_reward = torch.zeros((), device=agent.device)
    for _ in range(length):
        _, action = agent.get_logits_action(observation)
        observation, reward, done, _ = env.step(Agent.dist_postprocess(action))
        episodes += torch.sum(done)
        episode_reward += torch.sum(reward)
    return episodes, episode_reward / episodes


def train_unroll(agent, env, observation, num_unrolls, unroll_length):
    """Return step data over multple unrolls."""
    sd = StepData([], [], [], [], [], [])
    # print(f'train_unroll: {num_unrolls} {unroll_length}')
    for _ in range(num_unrolls):
        one_unroll = StepData([observation], [], [], [], [], [])
        for _ in range(unroll_length):
            logits, action = agent.get_logits_action(observation)
            observation, reward, done, info = env.step(Agent.dist_postprocess(action))
            # print(f'{observation.shape=}')
            one_unroll.observation.append(observation)
            one_unroll.logits.append(logits)
            one_unroll.action.append(action)
            one_unroll.reward.append(reward)
            one_unroll.done.append(done)
            one_unroll.truncation.append(info['truncation'])
        one_unroll = sd_map(torch.stack, one_unroll)
        # print(f'{one_unroll.observation.shape=}')
        sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
    td = sd_map(torch.stack, sd)
    return observation, td


def shuffle_and_batch(td, batch_size, device):
    with torch.no_grad():
        # drop the incomplete batch, if necessary
        # print(f'A {td.observation.shape[1]=}')
        if td.observation.shape[1] % batch_size != 0:
            td = sd_map(lambda d: d[:, :-(d.shape[1] % batch_size)], td)
        # print(f'B {td.observation.shape[1]=}')

        permutation = torch.randperm(
            td.observation.shape[1], device=device)

        def shuffle_batch(data):
            data = data[:, permutation]
            data = data.reshape([data.shape[0], -1, batch_size] +
                                list(data.shape[2:]))
            return data.swapaxes(0, 1)

        epoch_td = sd_map(shuffle_batch, td)

    num_minibatches = td.observation.shape[1] // batch_size
    assert td.observation.shape[1] % batch_size == 0
    for minibatch_i in range(num_minibatches):
        yield sd_map(lambda d: d[minibatch_i], epoch_td)