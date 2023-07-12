import torch
import numpy
import time

def gen_linspace(message, start, end, steps, log="always", log_config="fail"):
    _start = time.time()
    output = torch.linspace(start, end, steps)
    duration = time.time() - _start
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
    
def gen_randn(message, size, log="always", log_config="fail"):
    _start = time.time()
    output = torch.randn(size)
    duration = time.time() - _start
    return {
            "message": message,
            "func": "randn",
            "args": {
                "size": size
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
    }
    
def gen_uniform(message, size, log="always", log_config="fail"):
    _start = time.time()
    output = torch.rand(size)
    duration = time.time() - _start
    return {
            "message": message,
            "func": "uniform",
            "args": {
                "size": size
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
    }