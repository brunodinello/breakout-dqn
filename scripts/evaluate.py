"""Load a trained checkpoint and play Breakout.

Example:
    python scripts/evaluate.py --checkpoint checkpoints/double_dqn_final.dat --episodes 5
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dqn_cnn_model import DQN_CNN_Model  # noqa: E402
from utils import make_env  # noqa: E402


def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float32) / 255.0


def main():
    p = argparse.ArgumentParser(description="Run a trained agent on Breakout.")
    p.add_argument("--checkpoint", required=True, help="Path to .dat state_dict")
    p.add_argument("--env", default="ALE/Breakout-v5")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", action="store_true", help="Show a window while playing")
    p.add_argument("--video-dir", default=None, help="Save episode videos to this folder")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    env = make_env(
        args.env,
        render_mode="human" if args.render else "rgb_array",
        video_folder=args.video_dir,
        name_prefix="breakout_eval",
        record_every=1 if args.video_dir else None,
    )
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = DQN_CNN_Model(obs_shape, n_actions).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rewards = []
    for ep in range(args.episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            s = state_to_tensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(model(s).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
        print(f"Episode {ep + 1}: reward = {total}")

    print(f"\nMean reward over {args.episodes} episodes: {np.mean(rewards):.2f}")
    print(f"Best episode: {np.max(rewards):.2f}")


if __name__ == "__main__":
    main()
