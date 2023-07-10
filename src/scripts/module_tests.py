import torch
import numpy
import time
from model_loading import gen_state_dict

def gen_nn_multihead_attention(message, query, key, value, embed_dim, num_heads, log="always", log_config="fail"):
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads)
    start = time.time()
    output, weights = mha(query, key, value)
    duration = time.time() - start
    return {
        "message": message,
        "func": "nn_multihead_attention",
        "args": {
            "query": query.numpy().tolist(),
            "key": key.numpy().tolist(),
            "value": value.numpy().tolist(),
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "state_dict": gen_state_dict(mha)
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_nn_layernorm(message, input, norm_shape, log="always", log_config="fail"):
    ln = torch.nn.LayerNorm(norm_shape)
    start = time.time()
    output = ln(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "nn_layernorm",
        "args": {
            "input": input.numpy().tolist(),
            "norm_shape": norm_shape
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_nn_groupnorm(message, input, num_groups, num_channels, log="always", log_config="fail"):
    gn = torch.nn.GroupNorm(num_groups, num_channels)
    start = time.time()
    output = gn(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "nn_groupnorm",
        "args": {
            "input": input.numpy().tolist(),
            "num_groups": num_groups,
            "num_channels": num_channels,
            "state_dict": gen_state_dict(gn)
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_nn_linear(message, input, in_channels, out_channels, log="always", log_config="fail"):
    start = time.time()
    ln = torch.nn.Linear(in_channels, out_channels)
    output = ln(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "nn_linear",
        "args": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input": input.numpy().tolist(),
            "state_dict": gen_state_dict(ln)
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_nn_conv2d(message, input, in_channels, out_channels, kernel_size, log="always", log_config="fail"):
    start = time.time()
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
    output = conv(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "nn_conv2d",
        "args": {
            "input": input.numpy().tolist(),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "state_dict": gen_state_dict(conv)
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_nn_maxpool2d(message, input, kernel_size, log="always", log_config="fail"):
    start = time.time()
    conv = torch.nn.MaxPool2d(kernel_size)
    output = conv(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "nn_maxpool2d",
        "args": {
            "input": input.numpy().tolist(),
            "kernel_size": kernel_size
        },
        "target": output.numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }