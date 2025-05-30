import copy
import random
import time
from pathlib import Path

import jax
import math
import numpy as np
import torch
from tqdm import tqdm

from optimizer import get_optimizer
import torch.utils.tensorboard as tb

from task.ppo_torch import Agent, eval_unroll, train_unroll, sd_map, unroll_first, shuffle_and_batch
from search_space.cs import ConfigSpaceSearchSpace
from task import brax_task_utils
from utils.util_fns import adjust_optimizer_settings, convert_config_from_logarithmic

from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
import traceback


class BraxTask:
    def __init__(self, cfg, search_space, **__):
        self.cfg = cfg
        self.search_space: ConfigSpaceSearchSpace = search_space
        self.num_evals_per_step = self.cfg.task.num_evals_per_step # how often to eval for the curves
        self.viz = self.cfg.task.get('viz', False)

        # Hardcoded values from brax/BG-PBT:
        self.num_envs = self.cfg.task.get('num_envs', 2048)
        self.episode_length = 1000
        self.num_minibatches = 32

        self.env_name = self.cfg.task.name
        def _create_env():
            env = envs.create(self.env_name, batch_size=self.num_envs,
                              episode_length=self.episode_length,
                              backend='spring')
            env = gym_wrapper.VectorGymWrapper(env)
            env.seed(cfg.general.seed_base) # seeds different from base will be set in every __call__
            return env
        self.env = _create_env()
        self.env_eval = _create_env()


    def __call__(self, seed, solution, t, t_step, cpkt_loaded, tensorboard_dir, only_evaluate):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if only_evaluate is None:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = tb.SummaryWriter(tensorboard_dir)

        # create env - within __call__ so that the seed would influence it.
        env = self.env
        env.seed(seed)
        env = torch_wrapper.TorchWrapper(env, device=device)
        # Warm-start
        observation = env.reset()
        action = torch.zeros(env.action_space.shape).to(device)
        env.step(action)
        env_eval = torch_wrapper.TorchWrapper(self.env_eval, device=device)
        env_eval.seed(random.randint(0, 2**32 - 1))

        # get rl values
        rl_config = self.get_rl_vals(solution)
        batch_size = rl_config['batch_size']
        num_update_epochs = rl_config['num_update_epochs']
        unroll_length = rl_config['unroll_length']

        # create agent
        agent = self.create_agent(env, rl_config)
        agent.load_state_dict(cpkt_loaded['model_state_dict'])
        agent.to(device)

        if only_evaluate is not None:
            assert type(only_evaluate) == list
            out = {}

            if 'val' in only_evaluate:
                eval_res = self._eval(env_eval, agent, [], t, None)
                out['val'] = eval_res[0]
                out['fitness'] = eval_res[0]

            if 'test' in only_evaluate:
                env_eval.seed(seed) # needs to be reproducible
                eval_res = self._eval(env_eval, agent, [], t, None)
                out['test'] = eval_res[0]
                gif = self.visualize_policy(agent)
                out['policy_gif'] = gif

            return out

        # create optimizer
        solution_optimizer_vals = self.get_optimizer_vals(solution)
        optimizer = get_optimizer(self.cfg, agent, solution_optimizer_vals)
        optimizer.load_state_dict(cpkt_loaded['optimizer_state_dict']) # this will overwrite solution_optimizer_vals
        optimizer = adjust_optimizer_settings(optimizer, solution_optimizer_vals)
        del cpkt_loaded

        losses = {'train': [], 'test': []}
        curve = []

        # Should do t_step environment steps (and potentially more training steps).
        # To have the exact number of env steps, generate more and discard the excess. This is efficient thanks to brax.
        # Note that in order to have the desired number of intermediate evaluations,
        # we need to base them on training steps, not env steps.
        num_steps = batch_size * self.num_minibatches * unroll_length
        assert t_step >= num_steps
        num_epochs = t_step // num_steps
        num_unrolls = batch_size * self.num_minibatches // env.num_envs
        extra_unrolls_last_epoch = math.ceil((t_step % num_steps) / (env.num_envs * unroll_length)) # to go over
        excess_individual_unrolls = extra_unrolls_last_epoch * env.num_envs - math.ceil((t_step % num_steps)/unroll_length) # to go back to target
        steps_last_epoch = (num_unrolls + extra_unrolls_last_epoch) * env.num_envs - excess_individual_unrolls

        total_training_batches = (num_epochs - 1) * num_update_epochs * self.num_minibatches + \
                                 1 * num_update_epochs * \
                                 (steps_last_epoch // batch_size) # important to divide by batch_size before multiplication

        eval_every_batches = total_training_batches // self.num_evals_per_step
        batches_since_last_eval = 0
        num_evals_done = 0

        try:
            for i_inner_epoch in tqdm(range(num_epochs), desc=f'train+val'):
                agent.train()

                # train_unroll makes (num_unrolls * unroll_length * num_envs) steps of the environment.
                # num_unrolls is controlled by me. unroll_length is a searchable HP.
                # num_envs is controlled by me but should probably not be changed to keep efficiency high.
                observation, td = train_unroll(agent, env, observation,
                                               num_unrolls + (0 if i_inner_epoch != num_epochs - 1 else extra_unrolls_last_epoch),
                                               unroll_length)
                td = sd_map(unroll_first, td)

                if i_inner_epoch == num_epochs - 1 and extra_unrolls_last_epoch > 0:
                    if excess_individual_unrolls > 0:
                        def remove_extra(data):
                            return data[:, :-excess_individual_unrolls, ...]
                        td = sd_map(remove_extra, td)

                agent.update_normalization(td.observation)

                for i_update_epoch in range(num_update_epochs):
                    for i_batch, td_minibatch in enumerate(shuffle_and_batch(td, batch_size, device)):
                        loss = agent.loss(td_minibatch._asdict())
                        losses['train'].append(loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        batches_since_last_eval += 1
                        if batches_since_last_eval >= eval_every_batches:
                            num_evals_done += 1
                            timestamp = int(t + num_evals_done * (eval_every_batches / total_training_batches) * t_step) # fake timestamp, due to diff between env and training steps, see reasoning above
                            val_reward, curve = self._eval(env_eval, agent, curve, timestamp, tb_writer)
                            batches_since_last_eval = 0

                            tb_writer.add_scalar('loss/train', np.mean(losses['train']), timestamp)
                            losses['train'] = []

        except Exception as e:
            print(f'Exception in RL training: {e}')
            print(traceback.format_exc())
            val_reward = -10e9
            curve = []
            for i in range(self.num_evals_per_step):
                timestamp = int(t + i * (eval_every_batches / total_training_batches) * t_step)
                curve.append((timestamp, val_reward))


        dict_to_save = {'model_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}

        tb_writer.close()
        print(f'{curve=}')

        if self.viz:
            gif = self.visualize_policy(agent)
            with open(Path(self.cfg.path.dir_exp) / f'policy_{t:09d}.webp', 'wb') as f:
                f.write(gif)

        return {'fitness': curve[-1][1],
                'curve': curve,
                'dict_to_save': dict_to_save,
                'metrics': {'val': val_reward, 'test': None}}

    def create_agent(self, env, rl_config):
        policy_layers = [env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2]
        value_layers = [env.observation_space.shape[-1], 64, 64, 1]
        agent = Agent(policy_layers, value_layers, rl_config['entropy_cost'], rl_config['discounting'],
                      rl_config['reward_scaling'], rl_config['lambda_'], rl_config['ppo_epsilon'],
                      'cuda')
        return agent

    def _eval(self, env_eval, agent, curve, timestamp, tb_writer):
        st = time.time()
        agent.eval()
        with torch.no_grad():
            episode_count, episode_reward = eval_unroll(agent, env_eval, self.episode_length)
            episode_reward = episode_reward.item()

        if tb_writer is not None:
            tb_writer.add_scalar('reward/val', episode_reward, timestamp)

        curve.append((timestamp, episode_reward))
        agent.train()
        print(f'Eval time: {time.time() - st:.2f} seconds')
        return episode_reward, curve

    def get_optimizer_vals(self, solution):
        config_dict = self.search_space.vector_to_dict(solution)
        config_dict = convert_config_from_logarithmic(config_dict)
        config_dict = {k:v for k, v in config_dict.items() if k in ['lr', 'weight_decay', 'momentum']}
        return config_dict

    def get_rl_vals(self, solution):
        config = copy.deepcopy(PPO_DEFAULT_CONFIGS[self.env_name])

        config_dict = self.search_space.vector_to_dict(solution)
        config_dict = convert_config_from_logarithmic(config_dict)

        for k, v in config_dict.items():
            if k in config:
                config[k] = v

        return config


    def prepare_initial_ckpt(self, solution):
        rl_config = self.get_rl_vals(solution)
        model = self.create_agent(self.env, rl_config)
        optimizer = get_optimizer(self.cfg, model, self.get_optimizer_vals(solution))
        return {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

    def visualize_policy(self, agent):
        agent = agent.cuda()
        env = envs.get_environment(self.env_name, backend='spring')
        rollout = []
        state = env.reset(jax.random.PRNGKey(0))
        env_step_jit = jax.jit(env.step)

        import brax.io.torch as brax_torch
        for _ in tqdm(range(1000)):
            rollout.append(state.pipeline_state)

            _, act = agent.get_logits_action(brax_torch.jax_to_torch(state.obs))
            act = agent.dist_postprocess(act)

            state = env_step_jit(state, brax_torch.torch_to_jax(act))

            if state.done:
                break

        gif = brax_task_utils.render(env.sys, rollout, fmt='webp', camera_id=0 if self.env_name != 'pusher' else -1)
        return gif

# optimized values from brax (except for lambda_, ppo_epsilon)
# for hopper and those after it use the same unoptimized default values as BG-PBT. Also for humanoid.
PPO_DEFAULT_CONFIGS = {
    'ant': {
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'humanoid': { # values copied from hopper
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'halfcheetah': {
        'discounting': 0.95,
        'entropy_cost': 0.001,
        'unroll_length': 20,
        'reward_scaling': 1,
        'batch_size': 512,
        'num_update_epochs': 8,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'fetch': {
        'discounting': 0.997,
        'entropy_cost': 0.001,
        'unroll_length': 20,
        'reward_scaling': 5,
        'batch_size': 256,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'grasp': {
        'discounting': 0.99,
        'entropy_cost': 0.001,
        'unroll_length': 20,
        'reward_scaling': 10,
        'batch_size': 256,
        'num_update_epochs': 2,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'ur5e': {
        'discounting': 0.95,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'reacher': {
        'discounting': 0.95,
        'entropy_cost': 0.001,
        'unroll_length': 50,
        'reward_scaling': 5,
        'batch_size': 256,
        'num_update_epochs': 8,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'hopper': {
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'walker2d': {
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
    'pusher': {
        'discounting': 0.97,
        'entropy_cost': 0.01,
        'unroll_length': 5,
        'reward_scaling': 10,
        'batch_size': 1024,
        'num_update_epochs': 4,
        'lambda_': 0.95,
        'ppo_epsilon': 0.3,
    },
}
