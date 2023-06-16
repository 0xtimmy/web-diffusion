import torch;
import json;
import tests as t;

# Configure and build the generated test file

tests = [
    t.gen_unsqueeze("Unsqueeze test 1", [10], 0),
    t.gen_unsqueeze("Unsqueeze test 2", [10, 3, 2], 0),
    t.gen_unsqueeze("Unsqueeze test 3", [10, 3, 2], 1),
    t.gen_unsqueeze("Unsqueeze test 3", [10, 3, 2], -1),
]

def build():
    out = json.dumps(tests)
    fp = open("./tests.json", "w")
    fp.write(out)
    fp.close()

if __name__ == "__main__":
    build()