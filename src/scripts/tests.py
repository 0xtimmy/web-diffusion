
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