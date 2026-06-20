import numpy as np

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from gymnasium.wrappers import AtariPreprocessing
from gymnasium.spaces import Box

from .preprocess import _to_chw, get_img_preprocessor

ATARI_100K_GAMES = {
    "alien": "Alien",
    "amidar": "Amidar",
    "assault": "Assault",
    "asterix": "Asterix",
    "bank_heist": "BankHeist",
    "battle_zone": "BattleZone",
    "boxing": "Boxing",
    "breakout": "Breakout",
    "chopper_command": "ChopperCommand",
    "crazy_climber": "CrazyClimber",
    "demon_attack": "DemonAttack",
    "freeway": "Freeway",
    "frostbite": "Frostbite",
    "gopher": "Gopher",
    "hero": "Hero",
    "jamesbond": "Jamesbond",
    "kangaroo": "Kangaroo",
    "krull": "Krull",
    "kung_fu_master": "KungFuMaster",
    "ms_pacman": "MsPacman",
    "pong": "Pong",
    "private_eye": "PrivateEye",
    "qbert": "Qbert",
    "road_runner": "RoadRunner",
    "seaquest": "Seaquest",
    "up_n_down": "UpNDown",
}

class AtariEnv:
    """
    """
    def __init__(
            self,
            game,
            img_size,
            processed_img_size=224,
            action_repeat=4,
            noop_max=30,
            repeat_action_probability=0.0,
            terminal_on_life_loss=False,
            grayscale_obs=False,
            full_action_space=False,
            render_mode=None
    ):
        if game not in ATARI_100K_GAMES:
            raise ValueError(f"Unknown Atari-100K game '{game}'. Expected one of {sorted(ATARI_100K_GAMES)}")

        self.game = game
        self.img_size = img_size
        self.action_repeat = action_repeat
        self.noop_max = noop_max
        self.repeat_action_probability = repeat_action_probability
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.full_action_space = full_action_space
        env_id = f"ALE/{ATARI_100K_GAMES[game]}-v5"

        base = gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            render_mode=render_mode,
        )
        wrapper_kwargs = dict(
            noop_max=noop_max,
            frame_skip=action_repeat,
            screen_size=self.img_size[0],
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=grayscale_obs,
            scale_obs=False,
        )
        self.env = AtariPreprocessing(base, **wrapper_kwargs)
        self.action_space = self.env.action_space
        self.num_actions = self.action_space.n

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, processed_img_size, processed_img_size),
            dtype=np.float32,
        )

        self.episode_return = 0.0
        self.episode_length = 0

        self.preprocessor = get_img_preprocessor(source='obs', target='obs', img_size=processed_img_size)

    def _process_obs(self,obs):
        batch = {'obs': _to_chw(obs)}
        batch = self.preprocessor(batch)
        return batch['obs'].cpu().float().numpy()

    def reset(self, seed = None):
        """
        Returns observation, info
        """
        obs, info = self.env.reset(seed=seed)
        self.episode_return = 0.0
        self.episode_length = 0
        
        return self._process_obs(obs), info
    
    def step(self, action):
        """
        Returns the usual 5 values that a env.step() returns
        """
        obs, reward, terminated, truncated, info = self.env.step(int(action))

        self.episode_return += float(reward)
        self.episode_length += 1

        return self._process_obs(obs), reward, terminated, truncated, info
    
    def sample_action(self):
        return int(self.action_space.sample())

    def close(self):
        self.env.close()