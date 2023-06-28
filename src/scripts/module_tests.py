import torch
import numpy

def gen_nn_linear(message, in_channels, out_channels, input, log="always", log_config="fail"):
    start = time.tim()
    ln = torch.nn.Linear(in_channels, out_channels)
    output = ln(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "nn_linear",
        "args": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input": input.numpy().tolist()
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }