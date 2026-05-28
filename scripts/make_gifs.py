"""Build the README GIFs from the recorded training videos.

Requires `ffmpeg` on PATH. For each run folder (dqn_training/, ddqn_training/),
picks a set of milestone episodes, converts each mp4 to a small GIF, then
horizontally concatenates them into a single "progression" GIF.

Example:
    python scripts/make_gifs.py --run videos/ddqn_training --out assets/double_dqn_progression.gif
"""
import argparse
import os
import shutil
import subprocess
import tempfile


DEFAULT_EPISODES = [0, 2000, 5000, 9000]


def mp4_to_gif(mp4_path, gif_path, fps=15, width=240):
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos",
        "-loop", "0",
        gif_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def hstack_gifs(gif_paths, out_path):
    inputs = []
    for g in gif_paths:
        inputs += ["-i", g]
    n = len(gif_paths)
    filter_complex = "".join(f"[{i}:v]" for i in range(n)) + f"hstack=inputs={n}"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Folder with breakout-episode-N.mp4 files")
    p.add_argument("--out", required=True, help="Output GIF path")
    p.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    p.add_argument("--width", type=int, default=240)
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH. Install it (brew install ffmpeg).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        gif_paths = []
        for ep in args.episodes:
            mp4 = os.path.join(args.run, f"breakout-episode-{ep}.mp4")
            if not os.path.exists(mp4):
                print(f"skip: {mp4} not found")
                continue
            gif = os.path.join(tmp, f"ep{ep}.gif")
            print(f"converting episode {ep} → {gif}")
            mp4_to_gif(mp4, gif, fps=args.fps, width=args.width)
            gif_paths.append(gif)

        if not gif_paths:
            raise SystemExit("No episodes found to combine.")

        if len(gif_paths) == 1:
            shutil.copy(gif_paths[0], args.out)
        else:
            print(f"stacking {len(gif_paths)} gifs → {args.out}")
            hstack_gifs(gif_paths, args.out)

    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
