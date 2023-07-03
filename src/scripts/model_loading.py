import torch
import time
import numpy
import json

def gen_state_dict(model):
    state = model.state_dict()
    json_state = {}
    for param_tensor in state:
        json_state[param_tensor] = state[param_tensor].numpy().tolist()
        print(param_tensor, state[param_tensor].shape)
    return json_state

def export_state_dict(model, filename="paramters.json", log="always", log_config="fail"):
    start = time.time()
    
    out = gen_state_dict(json.dumps(model))
    fp = open(filename, "w")
    fp.write(out)
    fp.close()

    duration = time.time() - start
    print(f"Completed in {duration}ms")