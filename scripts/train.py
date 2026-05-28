"""Train a DQN or Double DQN agent on Atari Breakout.

Example:
    python scripts/train.py --agent double_dqn --episodes 10000
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dqn_agent import DQNAgent  # noqa: E402
from double_dqn_agent import DoubleDQNAgent  # noqa: E402
from dqn_cnn_model import DQN_CNN_Model  # noqa: E402
from utils import make_env  # noqa: E402


def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float32) / 255.0


def main():
    p = argparse.ArgumentParser(description="Train a DQN/Double DQN on Breakout.")
    p.add_argument("--agent", choices=["dqn", "double_dqn"], default="double_dqn")
    p.add_argument("--env", default="ALE/Breakout-v5")
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--max-steps", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--memory-size", type=int, default=100_000)
    p.add_argument("--epsilon-i", type=float, default=1.0)
    p.add_argument("--epsilon-f", type=float, default=0.1)
    p.add_argument("--epsilon-anneal-steps", type=int, default=1_000_000)
    p.add_argument("--sync-target", type=int, default=1000,
                   help="(double_dqn only) steps between target-net syncs")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--video-dir", default=None,
                   help="If set, record an episode video every --record-every episodes")
    p.add_argument("--record-every", type=int, default=500)
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-project", default="breakout-dqn")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    env = make_env(
        args.env,
        video_folder=args.video_dir,
        name_prefix="breakout",
        record_every=args.record_every if args.video_dir else None,
    )
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    if args.agent == "dqn":
        model = DQN_CNN_Model(obs_shape, n_actions)
        agent = DQNAgent(
            env, model, state_to_tensor,
            memory_buffer_size=args.memory_size,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon_i=args.epsilon_i,
            epsilon_f=args.epsilon_f,
            epsilon_anneal_steps=args.epsilon_anneal_steps,
            episode_block=100,
            device=device,
        )
    else:
        model_a = DQN_CNN_Model(obs_shape, n_actions)
        model_b = DQN_CNN_Model(obs_shape, n_actions)
        model_b.load_state_dict(model_a.state_dict())
        agent = DoubleDQNAgent(
            env, model_a, model_b, state_to_tensor,
            memory_buffer_size=args.memory_size,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon_i=args.epsilon_i,
            epsilon_f=args.epsilon_f,
            epsilon_anneal_steps=args.epsilon_anneal_steps,
            episode_block=100,
            device=device,
            sync_target=args.sync_target,
        )

    agent.checkpoint_dir = args.checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    agent.train(number_episodes=args.episodes, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
