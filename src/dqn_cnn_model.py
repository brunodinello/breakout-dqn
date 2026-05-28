import torch.nn as nn
import torch.nn.functional as F


def conv2d_output_shape(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """Return (H_out, W_out) for a Conv2d layer — useful to size the FC head."""
    def to_tuple(x):
        return (x, x) if isinstance(x, int) else x

    H_in, W_in = input_size
    ker_h, ker_w = to_tuple(kernel_size)
    str_h, str_w = to_tuple(stride)
    pad_h, pad_w = to_tuple(padding)
    dil_h, dil_w = to_tuple(dilation)

    H_out = (H_in + 2 * pad_h - dil_h * (ker_h - 1) - 1) // str_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (ker_w - 1) - 1) // str_w + 1
    return H_out, W_out


class DQN_CNN_Model(nn.Module):
    """Convolutional Q-network for Atari (Mnih et al., 2015 — smaller variant).

    Two conv layers + one FC layer + linear head over n_actions. Operates on
    stacks of (C, 84, 84) frames where C is the number of stacked frames.
    """

    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape

        self.conv1 = nn.Conv2d(c, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        h1, w1 = conv2d_output_shape((h, w), kernel_size=8, stride=4)
        h2, w2 = conv2d_output_shape((h1, w1), kernel_size=4, stride=2)
        conv_output_dim = 32 * h2 * w2

        self.fc1 = nn.Linear(conv_output_dim, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
