import gymnasium
import matplotlib.pyplot as plt
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordVideo,
    TransformReward,
)


def show_observation(observation):
    """Display a single frame (RGB, grayscale-3D, or 2D) with matplotlib."""
    dimension = observation.shape
    if len(dimension) == 3:
        if dimension[2] == 3:
            plt.imshow(observation)
        elif dimension[2] == 1:
            plt.imshow(observation[:, :, 0], cmap='gray')
    elif len(dimension) == 2:
        plt.imshow(observation, cmap='gray')
    else:
        raise ValueError("Invalid observation shape")
    plt.show()


def show_observation_stack(observation):
    """Display each frame of a stacked observation in sequence."""
    for i in range(observation.shape[0]):
        show_observation(observation[i])


class FireOnLifeLostWrapper(gymnasium.Wrapper):
    """Auto-press FIRE after reset and after every life lost.

    Breakout requires FIRE to launch the ball; without this wrapper the agent
    must learn to press it itself, which slows learning considerably.
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            return self.reset(**kwargs)
        self._prev_lives = info.get('lives')
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = info.get('lives', self._prev_lives)
        if current_lives is not None and self._prev_lives is not None \
                and current_lives < self._prev_lives and not (terminated or truncated):
            obs, fire_reward, terminated, truncated, info = self.env.step(1)
            reward += fire_reward
        self._prev_lives = current_lives
        return obs, reward, terminated, truncated, info


def make_env(
    env_name: str,
    render_mode: str = "rgb_array",
    video_folder: str | None = None,
    name_prefix: str = "",
    record_every: int | None = None,
    grayscale: bool = True,
    screen_size: int = 84,
    stack_frames: int = 4,
    skip_frames: int = 4,
) -> gymnasium.Env:
    """Build an Atari env with standard DQN preprocessing.

    - Frame skipping & noop reset (AtariPreprocessing)
    - Grayscale + resize to 84×84
    - Frame stacking
    - Reward clipping to {-1, 0, 1}
    - Optional video recording every `record_every` episodes
    """
    env = gymnasium.make(env_name, render_mode=render_mode, frameskip=1)

    if video_folder is not None and record_every is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep % record_every == 0,
            fps=env.metadata.get("render_fps", 30) * skip_frames,
        )

    env = AtariPreprocessing(
        env,
        noop_max=10,
        frame_skip=skip_frames,
        screen_size=screen_size,
        grayscale_obs=grayscale,
        grayscale_newaxis=False,
    )
    env = FrameStackObservation(env, stack_size=stack_frames)
    env = TransformReward(env, lambda r: 1 if r > 0 else (-1 if r < 0 else 0))
    return env
