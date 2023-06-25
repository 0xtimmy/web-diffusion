import torch
import torch.nn.functional as F
import time

def gen_scalar_add(message, input, alpha, log="always", log_config="fail"):
    start = time.time()
    output = input + alpha
    duration = time.time() - duration
    return {
        "message": message,
        "func": "scalar_add",
        "args": {
            "input": input.numpy().tolist(),
            "alpha": alpha,
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_scalar_sub(message, input, alpha, log="always", log_config="fail"):
    start = time.time()
    output = input - alpha
    duration = time.time() - duration
    return {
        "message": message,
        "func": "scalar_sub",
        "args": {
            "input": input.numpy().tolist(),
            "alpha": alpha,
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_scalar_mul(message, input, alpha, log="always", log_config="fail"):
    start = time.time()
    output = input * alpha
    duration = time.time() - duration
    return {
        "message": message,
        "func": "scalar_mul",
        "args": {
            "input": input.numpy().tolist(),
            "alpha": alpha,
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_scalar_div(message, input, alpha, log="always", log_config="fail"):
    start = time.time()
    output = input / alpha
    duration = time.time() - duration
    return {
        "message": message,
        "func": "scalar_div",
        "args": {
            "input": input.numpy().tolist(),
            "alpha": alpha,
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_sum(message, input, log="always", log_config="fail"):
    start = time.time()
    output = input.sum()
    duration = time.time() - start
    return {
            "message": message,
            "func": "sum",
            "args": {
                "input": input.numpy().tolist(),
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
        }

def gen_mm(message, input, weight, log="always", log_config="fail"):
    start = time.time()
    output = input.mm(weight)
    duration = time.time() - start
    return {
        "message": message,
        "func": "mm",
        "args": {
            "input": input.numpy().tolist(),
            "weight": weight.numpy().tolist(),
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }