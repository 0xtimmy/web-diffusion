import torch;
import json;
import tests as t;

# Configure and build the generated test file

tests = [
    t.gen_conv2d("Conv2d test 1", torch.randn(1, 4, 5, 5), torch.randn(8, 4, 5, 5), None),
    t.gen_conv2d("Conv2d test 2", torch.randn(1, 4, 5, 5), torch.randn(8, 4, 5, 5), None),
    t.gen_conv2d("Conv2d test 3 - image dimensions", torch.randn(1, 3, 64, 64), torch.randn(3, 3, 64, 64), None),
    t.gen_transpose("Transpose test 1 - 3 x 2", torch.randint(1, 10, [3, 2])),
    t.gen_transpose("Transpose test 2 - transpose dims: 1 & 2 for matrix: 3 x 2 x 2", torch.randint(1, 10, [3, 2, 2]), 1, 2),
    t.gen_transpose("Transpose test 2 - transpose dims: 0 & 2 for matrix: 3 x 2 x 2", torch.randint(1, 10, [3, 2, 2]), 0, 2),
    t.gen_transpose("Transpose test 2 - transpose dims: 0 & 2 for matrix: 2 x 3 x 2", torch.randint(1, 10, [2, 3, 2]), 0, 2),
    t.gen_transpose("Transpose test 2 - transpose dims: 0 & 2 for matrix: 2 x 2 x 3", torch.randint(1, 10, [2, 2, 3]), 0, 2),
    t.gen_linspace("Linspace test 1", 1, 10, 10),
    t.gen_linspace("Linspace test 2", 1, 10, 100),
    t.gen_linspace("Linspace test 3", 0, 1, 50),
    t.gen_linear("Linear test 1 - small no bias", torch.randint(1, 5, [3]), torch.randint(1, 5, [2, 3])),
    t.gen_linear("Linear test 2 - small with bias", torch.randint(1, 5, [3]), torch.randint(1, 5, [2, 3]), torch.randint(0, 1, [2])),
    t.gen_linear("Linear test 3 - default", torch.randint(1, 5, [20]), torch.randint(1, 5, [50, 20]), torch.randint(1, 5, [50])),
    t.gen_linear("Linear test 4 - no bias", torch.randint(1, 5, [20]), torch.randint(1, 5, [50, 20])),
    t.gen_linear("Linear test 5 - special zeros", torch.tensor([1, 1]), torch.tensor([[0, 0], [1, 1]])),
    t.gen_mm("mm test 1 - 2 x 2", torch.randint(1, 5, [2, 2]), torch.randint(1, 5, [2, 2])),
    t.gen_mm("mm test 1 - 3 x 3", torch.randint(1, 5, [3, 3]), torch.randint(1, 5, [3, 3])),
    t.gen_mm("mm test 1 - 2 x 3", torch.randint(1, 5, [2, 3]), torch.randint(1, 5, [3, 2])),
    t.gen_mm("mm test 1 - 2 x 3", torch.randint(1, 5, [1, 3]), torch.randint(1, 5, [3, 2])),
    t.gen_unsqueeze("Unsqueeze test 1", [10], 0),
    t.gen_unsqueeze("Unsqueeze test 2", [10, 3, 2], 0),
    t.gen_unsqueeze("Unsqueeze test 3", [10, 3, 2], 1),
    t.gen_unsqueeze("Unsqueeze test 3", [10, 3, 2], -1),
    t.gen_squeeze("Squeeze test 1", [10], 0),
    t.gen_squeeze("Squeeze test 2", [10, 1, 2, 1], None),
    t.gen_squeeze("Squeeze test 3", [10, 1, 2, 1], 1),
    t.gen_squeeze("Squeeze test 3", [10, 1, 2, 1], -1),
]

def build():
    out = json.dumps(tests)
    fp = open("./tests.json", "w")
    fp.write(out)
    fp.close()

if __name__ == "__main__":
    build()