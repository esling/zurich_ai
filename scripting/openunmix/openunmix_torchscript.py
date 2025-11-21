import os
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.mobile_optimizer import optimize_for_mobile

from openunmix import utils
from openunmix import model
from openunmix.model import OpenUnmix

target_urls_umxhq = {
    "bass": "bass_s.pth",
    "drums": "drums_s.pth",
    "other": "other_s.pth",
    "vocals": "vocals_s.pth",
}

target_urls_umxl = {
    "bass": "bass.pth",
    "drums": "drums.pth",
    "other": "other.pth",
    "vocals": "vocals.pth",
}


def get_umx_models(
    target_urls, hidden_size=512, targets=None, device="cpu", pretrained=True
):
    """Download openunmix pretrained models

    Args:
        target_urls: dict with the link to download the model for bass, drums, other and vocals
        hidden_size: size for bottleneck layer
        targets: list of stems
        device: the device on which the model will be used
        pretrained: boolean for pretrained weights

    Returns:
        target_models: list with all the models
    """
    if targets is None:
        targets = ["vocals", "drums", "bass", "other"]

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = int(utils.bandwidth_to_max_bin(rate=44100.0, n_fft=4096, bandwidth=16000))

    target_models = {}
    for target in targets:
        # load open unmix model
        target_unmix = OpenUnmix(
            nb_bins=4096 // 2 + 1,
            nb_channels=2,
            hidden_size=hidden_size,
            max_bin=max_bin,
        )

        # enable centering of stft to minimize reconstruction error
        if pretrained:
            state_dict = torch.load(
                target_urls[target], map_location=device
            )
            target_unmix.load_state_dict(state_dict, strict=False)
            target_unmix.eval()

        target_unmix.to(device)
        target_models[target] = target_unmix
    return target_models


def create_separator(target_models, device="cpu"):
    """Create separator class which contains all models

    Args:
        target_models: list of all models
        device: the device on which the model will be used

    Returns:
        separator: separator class which contains all models
    """
    separator = (
        model.Separator(
            target_models=target_models,
            niter=1,
            residual=False,
            n_fft=4096,
            n_hop=1024,
            nb_channels=2,
            sample_rate=44100.0,
            filterbank="asteroid",
        )
        .eval()
        .to(device)
    )

    return separator

def quantize_model(model):
    """Quantize model dynamically

    Args:
        model: model corresponding to the separator
    """
    model = torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )

    return model

def create_script(model_name, separator):
    """Create the torchscript model from a separator

    Args:
        model_name: name of the torchscript file to create
        separator: separator class which contains all models
    """
    jit_script = torch.jit.script(separator)
    torch.jit.save(jit_script, "unmix.pt")


def main():
    device = "cpu"

    separator_umxl = create_separator(
        get_umx_models(target_urls_umxhq, hidden_size=1024), device=device
    )

    if not os.path.exists("dist"):
        os.mkdir("dist")

    #separator_umxl = quantize_model(separator_umxl)

    create_script("umxhq", separator_umxl)


if __name__ == "__main__":
    main()
