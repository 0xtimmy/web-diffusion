import torch
import torch.nn.functional as F
import numpy

# Each gen function should return a json object of the below form
# These will be passed to tester.ts and run against our ts-torch

#interface test {
#    "message": string,         // the message to go along with the test
#    "func": string,            // the function being tested
#    "args": any,               // the arguements to be passed to the function
#    "target": any,             // the desired output
#    "log"?: logconfig,         // whether to log the test or not
#    "log_config"?: logconfig,  // whether to log the test arguements and output
#}

def gen_chunk(message, input, chunks, dim, log="always", log_config="fail"):
    output = torch.chunk(input, chunks, dim)
    return {
        "message": message,
        "func": "chunk",
        "args": {
            "input": input.numpy().tolist(),
            "chunks": chunks,
            "dim": dim,
        },
        "target": [t.numpy().tolist() for t in output],
        "log": log,
        "log_config": log_config
    }

def gen_layer_norm(message, input, norm_shape, weight, bias, log="always", log_config="fail"):
    output = torch.layer_norm(input, norm_shape, weight, bias)
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
        "log": log,
        "log_config": log_config
    }

def gen_group_norm(message, input, groups, weight, bias, log="always", log_config="fail"):
    output = torch.group_norm(input, groups, weight, bias)
    return {
        "message": message,
        "func": "group_norm",
        "args": {
            "input": input.numpy().tolist(),
            "groups": groups,
            "weight": weight.numpy().tolist(),
            "bias": weight.numpy().tolist()
        },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }


def gen_scalar_add(message, input, alpha, log="always", log_config="fail"):
    output = input + alpha
    return {
        "message": message,
        "func": "scalar_add",
        "args": {
            "input": input.numpy().tolist(),
            "alpha": alpha,
        },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_scalar_sub(message, input, alpha, log="always", log_config="fail"):
    output = input - alpha
    return {
        "message": message,
        "func": "scalar_sub",
        "args": {
            "input": input.numpy().tolist(),
            "alpha": alpha,
        },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_scalar_mul(message, input, alpha, log="always", log_config="fail"):
    output = input * alpha
    return {
        "message": message,
        "func": "scalar_mul",
        "args": {
            "input": input.numpy().tolist(),
            "alpha": alpha,
        },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_scalar_div(message, input, alpha, log="always", log_config="fail"):
    output = input / alpha
    return {
        "message": message,
        "func": "scalar_div",
        "args": {
            "input": input.numpy().tolist(),
            "alpha": alpha,
        },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_sum(message, input, log="always", log_config="fail"):
    output = input.sum()
    return {
            "message": message,
            "func": "sum",
            "args": {
                "input": input.numpy().tolist(),
            },
            "target": output.numpy().tolist(),
            "log": log,
            "log_config": log_config
        }

def gen_scaled_dot_product_attention(message, query, key, value, log="always", log_config="fail"):
    output = F.scaled_dot_product_attention(query, key, value)
    return {
            "message": message,
            "func": "scaled_dot_product_attention",
            "args": {
                "query": query.numpy().tolist(),
                "key": key.numpy().tolist(),
                "value": value.numpy().tolist(),
            },
            "target": output.numpy().tolist(),
            "log": log,
            "log_config": log_config
        }

def gen_max_pool2d(message, input, kernelSize, log="always", log_config="fail"):
    output = torch.max_pool2d(input, kernelSize)
    return {
            "message": message,
            "func": "max_pool2d",
            "args": {
                "input": input.numpy().tolist(),
                "kernelSize": kernelSize,
            },
            "target": output.numpy().tolist(),
            "log": log,
            "log_config": log_config
        }

def gen_conv2d(message, input, weight, bias, log="always", log_config="fail"):
    output = torch.conv2d(input, weight, bias)
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
            "log": log,
            "log_config": log_config
        }

def gen_linspace(message, start, end, steps, log="always", log_config="fail"):
    output = torch.linspace(start, end, steps)
    return {
            "message": message,
            "func": "linspace",
            "args": {
                "start": start,
                "end": end,
                "steps": steps
            },
            "target": output.numpy().tolist(),
            "log": log,
            "log_config": log_config
    }


def gen_transpose(message, input, dim0=0, dim1=1, log="always", log_config="fail"):
    output = input.transpose(dim0, dim1)
    return {
        "message": message,
        "func": "transpose",
        "args": {
            "input": input.numpy().tolist(),
            "dim0": dim0,
            "dim1": dim1,
        },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_mm(message, input, weight, log="always", log_config="fail"):
    output = input.mm(weight)
    return {
        "message": message,
        "func": "mm",
        "args": {
            "input": input.numpy().tolist(),
            "weight": weight.numpy().tolist(),
        },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_nn_linear(message, in_channels, out_channels, input, log="always", log_config="fail"):
    ln = torch.nn.Linear(in_channels, out_channels)
    output = ln(input)
    return {
        "message": message,
        "func": "nn_linear",
        "args": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input": input.numpy().tolist()
        },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_linear(message, input, weight, bias=None, log="always", log_config="fail"):
    output = torch.nn.functional.linear(input, weight, bias)
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
            "log": log,
            "log_config": log_config
        }
    

def gen_unsqueeze(message, shape, dim, log="always", log_config="fail"):
    input = torch.randint(0, 9, shape)
    output = input.unsqueeze(dim)
    return {
        "message": message,
        "func": "unsqueeze",
        "args": { "input": input.numpy().tolist(), "dim": dim },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_squeeze(message, shape, dim=None, log="always", log_config="fail"):
    input = torch.randint(0, 9, shape)
    if dim : 
        output = input.squeeze(dim)
    else :
        output = input.squeeze()
    return {
        "message": message,
        "func": "squeeze",
        "args": { "input": input.numpy().tolist(), "dim": dim },
        "target": output.numpy().tolist(),
        "log": log,
        "log_config": log_config
    }
