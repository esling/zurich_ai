from typing import List, Tuple

import os
import math
import torch
import torch.nn as nn
import torch
import nn_tilde

class Demucs(nn_tilde.Module):

    def __init__(self, 
                 pretrained):
        super().__init__()
        # REGISTER ATTRIBUTES
        self.register_attribute('sr', 44100)
        self.pretrained = pretrained

        # REGISTER METHODS
        self.register_method(
            'forward',
            in_channels=1,
            in_ratio=1,
            out_channels=4,
            out_ratio=1,
            input_labels=['(signal) signal to monitor'],
            output_labels=['drums', 'bass', 'vocals', 'others'],
        )

    @torch.jit.export
    def forward(self, input: torch.Tensor):
        in_r = input[0]
        out = self.pretrained(in_r).reshape(1, 4, -1).repeat(input.shape[0], 1, 1)
        return out

    # defining attribute getters
    # WARNING : typing the function's ouptut is mandatory
    @torch.jit.export
    def get_sr(self) -> int:
        return int(self.sr[0])

    # defining attribute setter
    # setters must return an error code :
    # return 0 if the attribute has been adequately set,
    # return -1 if the attribute was wrong.
    @torch.jit.export
    def set_sr(self, x: int) -> int:
        self.sr = (x, )
        return 0


if __name__ == '__main__':
    pretrained = torch.jit.load("demucs.pt")  # Pretrained weights
    model = Demucs(pretrained)
    model.export_to_ts('demucs.ts')