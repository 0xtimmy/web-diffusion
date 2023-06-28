import torch
import numpy

def gen_linspace(message, start, end, steps, log="always", log_config="fail"):
    start = time.time()
    output = torch.linspace(start, end, steps)
    duration = time.time() - start
    return {
            "message": message,
            "func": "linspace",
            "args": {
                "start": start,
                "end": end,
                "steps": steps
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
    }