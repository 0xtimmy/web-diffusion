import torch
import torch.nn.functional as F
import numpy
import time
import math

def gen_cumprod(message, input, dim=0, log="always", log_config="fail"):
    start = time.time()
    output = torch.cumprod(input, dim)
    duration = time.time() - start
    return {
        "message": message,
        "func": "cumprod",
        "args": {
            "input": input.numpy().tolist(),
            "dim": dim,
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_upsample(message, input, scale_factor=None, size=None, mode="nearest", log="always", log_config="fail"):
    start = time.time()
    output = F.interpolate(input, scale_factor=scale_factor, size=size, mode=mode)
    duration = time.time() - start
    return {
        "message": message,
        "func": "upsample",
        "args": {
            "input": input.numpy().tolist(),
            "scale_factor": scale_factor,
            "size": size,
            "mode": mode
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_softmax(message, input, dim, log="always", log_config="fail"):
    start = time.time()
    output = torch.softmax(input, dim)
    duration = time.time() - start
    return {
        "message": message,
        "func": "softmax",
        "args": {
            "input": input.numpy().tolist(),
            "dim": dim
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_silu(message, input, log="always", log_config="fail"):
    start = time.time()
    output = F.silu(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "silu",
        "args": {
            "input": input.numpy().tolist(),
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_gelu(message, input, log="always", log_config="fail"):
    start = time.time()
    output = F.gelu(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "gelu",
        "args": {
            "input": input.numpy().tolist(),
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_clamp(message, input, low, high, log="always", log_config="fail"):
    start = time.time()
    output = torch.clamp(input, low, high)
    duration = time.time() - start
    return {
        "message": message,
        "func": "clamp",
        "args": {
            "input": input.numpy().tolist(),
            "low": low,
            "high": high,
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }


def gen_layer_norm(message, input, norm_shape, weight, bias, log="always", log_config="fail"):
    start = time.time()
    output = torch.layer_norm(input, norm_shape, weight, bias)
    duration = time.time() - start
    return {
        "message": message,
        "func": "layer_norm",
        "args": {
            "input": input.numpy().tolist(),
            "norm_shape": norm_shape,
            "weight": weight.numpy().tolist(),
            "bias": weight.numpy().tolist()
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_group_norm(message, input, groups, weight, bias, log="always", log_config="fail"):
    start = time.time()
    output = torch.group_norm(input, groups, weight, bias)
    duration = time.time() - start
    return {
        "message": message,
        "func": "group_norm",
        "args": {
            "input": input.numpy().tolist(),
            "groups": groups,
            "weight": weight.numpy().tolist(),
            "bias": bias.numpy().tolist()
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }


def gen_scaled_dot_product_attention(message, query, key, value, log="always", log_config="fail"):
    start = time.time()
    output = F.scaled_dot_product_attention(query, key, value)
    duration = time.time() - start
    return {
            "message": message,
            "func": "scaled_dot_product_attention",
            "args": {
                "query": query.numpy().tolist(),
                "key": key.numpy().tolist(),
                "value": value.numpy().tolist(),
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
        }

def gen_max_pool2d(message, input, kernelSize, log="always", log_config="fail"):
    start = time.time()
    output = torch.max_pool2d(input, kernelSize)
    duration = time.time() - start
    return {
            "message": message,
            "func": "max_pool2d",
            "args": {
                "input": input.numpy().tolist(),
                "kernelSize": kernelSize,
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
        }

def gen_conv2d(message, input, weight, bias, log="always", log_config="fail"):
    start = time.time()
    output = torch.conv2d(input, weight, bias)
    duration = time.time() - start
    if bias:
        return {
            "message": message,
            "func": "conv2d",
            "args": {
                "input": input.numpy().tolist(),
                "weight": weight.numpy().tolist(),
                "bias": bias.numpy().tolist()
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
        }
    else:
        return {
            "message": message,
            "func": "conv2d",
            "args": {
                "input": input.numpy().tolist(),
                "weight": weight.numpy().tolist(),
                "bias": None
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
        }

def gen_linear(message, input, weight, bias=None, log="always", log_config="fail"):
    start = time.time()
    output = torch.nn.functional.linear(input, weight, bias)
    duration = time.time() - start
    if(bias is not None):
        return {
            "message": message,
            "func": "linear",
            "args": {
                "input": input.numpy().tolist(),
                "weight": weight.numpy().tolist(),
                "bias": bias.numpy().tolist()
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
        }
    return {
            "message": message,
            "func": "linear",
            "args": {
                "input": input.numpy().tolist(),
                "weight": weight.numpy().tolist(),
                "bias": None
            },
            "target": output.numpy().tolist(),
            "duration": duration * 1000,
            "log": log,
            "log_config": log_config
        }