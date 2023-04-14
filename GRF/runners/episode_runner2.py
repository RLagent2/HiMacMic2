from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.flag = 0
        self.won = 0.0

    def setup(self, scheme1, scheme2, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme1, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.new_skip_batch = partial(EpisodeBatch, scheme2, groups, self.batch_size, self.episode_limit + 1,
                                      preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.skip_batch = self.new_skip_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            actions, skip_n, temp_input = self.mac.select_together(self.batch, t_ep=self.t, t_env=self.t_env,
                                                                   test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()

            # actions,skip_n = self.mac.select_together(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            skip_states, skip_rewards, skip_obs = [], [], []
            # repeat the selected action for "skip" times
            skip_count = 0
            for curr_skip in range(skip_n + 1):
                skip_count += 1
                s = self.env.get_state()
                o = self.env.get_obs()
                av = self.env.get_avail_actions()

                pre_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()]
                }
                self.batch.update(pre_transition_data, ts=self.t)

                if curr_skip > 0:
                    actions = self.mac.select_with_input(temp_input, av, t_env=self.t_env, test_mode=test_mode)

                reward, terminated, env_info = self.env.step(actions[0])

                episode_return += reward

                post_transition_data = {
                    "actions": cpu_actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }
                # Update the replay buffer
                self.batch.update(post_transition_data, ts=self.t)

                skip_states.append(s)
                skip_rewards.append(reward)
                skip_obs.append(o)
                # Update the skip replay buffer with all observed skips.
                skip_id = 0
                for start_state, start_obs in zip(skip_states, skip_obs):
                    skip_reward = 0
                    for exp, r in enumerate(skip_rewards[skip_id:]):  # make sure to properly discount
                        skip_reward += np.power(0.95, exp) * r  # GAMMA = 0.95


                    skip_post_transition_data = {
                        "behaviour_action": cpu_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                        "length": [(curr_skip - skip_id + 1,)],
                        "action_idx": [(curr_skip - skip_id,)],
                        "state": start_state,
                        "obs": start_obs,
                        "actions": cpu_actions,

                    }
                    skip_id += 1
                    self.skip_batch.update(skip_post_transition_data, ts=self.t)

                self.t += 1
                if terminated:
                    break
            r_skip_k = 0
            for exp, r in enumerate(skip_rewards):
                r_skip_k += np.power(0.95, exp) * r
            r_i =r_skip_k / skip_count

            for ha in range(skip_count):
                post_reward_data = {
                    "int_reward": [(r_i,)],
                }
                # Update the replay buffer
                self.batch.update(post_reward_data, ts=self.t-skip_count+ha)


        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],

        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, skip_n, _ = self.mac.select_together(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        last_skip = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "behaviour_action": actions.to("cpu").numpy(),
            "length": [(1,)]
        }
        self.skip_batch.update(last_skip, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        self.flag = 0

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, self.skip_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
