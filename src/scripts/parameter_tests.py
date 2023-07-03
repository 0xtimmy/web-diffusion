import torch
import numpy
import time
from model_loading import gen_state_dict

def gen_linear_model_loading(message, in_channels, out_channels, input, log="always", log_config="fail"):
    ln = torch.nn.Linear(in_channels, out_channels)
    output = ln(input)
    start = time.time()
    state = gen_state_dict(ln)
    duration = time.time() - start

    return {
        "message": message,
        "func": "linear_model_loading",
        "args": {
            "input": input.numpy().tolist(),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "state_dict": state
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_compound_model_loading(message, in_channels, mid_channels, out_channels, input, log="always", log_config="fail"):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_channels, mid_channels),
        torch.nn.Linear(mid_channels, out_channels)
    )
    output = model(input)
    start = time.time()
    state = gen_state_dict(model)
    duration = time.time() - start

    return {
        "message": message,
        "func": "compound_model_loading",
        "args": {
            "input": input.numpy().tolist(),
            "in_channels": in_channels,
            "mid_channels": mid_channels,
            "out_channels": out_channels,
            "state_dict": state
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

