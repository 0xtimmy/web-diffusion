import torch;
import json;
import tests as t;

# Configure and build the generated test file

tests = [

    # Arithmetic Tests --------------------------------------------------------

    #t.gen_scalar_add("Scalar add basic test", torch.ones(2, 2), 1),
    #t.gen_scalar_add("Scalar sub basic test", torch.ones(2, 2), 1),
    #t.gen_scalar_add("Scalar mul basic test", torch.ones(2, 2), 2),
    #t.gen_scalar_add("Scalar div basic test", torch.ones(2, 2), 2),

    #t.gen_mm("mm test 1 - 2 x 2", torch.randint(1, 5, [2, 2]), torch.randint(1, 5, [2, 2])),
    #t.gen_mm("mm test 1 - 3 x 3", torch.randint(1, 5, [3, 3]), torch.randint(1, 5, [3, 3])),
    #t.gen_mm("mm test 1 - 2 x 3", torch.randint(1, 5, [2, 3]), torch.randint(1, 5, [3, 2])),
    #t.gen_mm("mm test 1 - 2 x 3", torch.randint(1, 5, [1, 3]), torch.randint(1, 5, [3, 2])),

    #t.gen_sum("Sum test small", torch.ones(2, 2)),
    #t.gen_sum("Sum test 100x100", torch.ones(100, 100)),
    #t.gen_sum("Sum test 1000x1000", torch.ones(1000, 1000)),


    # Functional Tests --------------------------------------------------------


    #t.gen_linear("Linear test 1 - small no bias", torch.randint(1, 5, [3]), torch.randint(1, 5, [2, 3])),
    #t.gen_linear("Linear test 2 - small with bias", torch.randint(1, 5, [3]), torch.randint(1, 5, [2, 3]), torch.randint(0, 1, [2])),
    #t.gen_linear("Linear test 3 - default", torch.randint(1, 5, [20]), torch.randint(1, 5, [50, 20]), torch.randint(1, 5, [50])),
    #t.gen_linear("Linear test 4 - no bias", torch.randint(1, 5, [20]), torch.randint(1, 5, [50, 20])),
    #t.gen_linear("Linear test 5 - special zeros", torch.tensor([1, 1]), torch.tensor([[0, 0], [1, 1]])),

    #t.gen_layer_norm("layer_norm basic test", torch.randn(20, 5, 10, 10), [5, 10, 10], torch.ones([5, 10, 10]), torch.zeros([5, 10, 10])),

    #t.gen_group_norm("group_norm basic test", torch.randn(20, 6, 10, 10), 2, torch.ones(6), torch.zeros(6)),

    #t.gen_scaled_dot_product_attention("Scaled dot product attention test 1", torch.rand(32, 8, 128, 64), torch.rand(32, 8, 128, 64), torch.rand(32, 8, 128, 64)),

    #t.gen_conv2d("Conv2d test 1", torch.randn(1, 4, 5, 5), torch.randn(8, 4, 5, 5), None),
    #t.gen_conv2d("Conv2d test 2", torch.randn(1, 4, 5, 5), torch.randn(8, 4, 5, 5), None),
    #t.gen_conv2d("Conv2d test 3 - image dimensions", torch.randn(1, 3, 64, 64), torch.randn(3, 3, 64, 64), None),

    #t.gen_max_pool2d("MaxPool2D test 1", torch.randn(4, 16, 16), 2),


    # Module Tests ------------------------------------------------------------


    # Reshape Tests -----------------------------------------------------------

    #t.gen_chunk("chunk test 1 - dim start", torch.randint(0, 10, [3, 5, 5]), 3, 0),
    #t.gen_chunk("chunk test 1 - dim mid", torch.randint(0, 10, [5, 3, 5]), 3, 1),
    #t.gen_chunk("chunk test 1 - dim end", torch.randint(0, 10, [5, 5, 3]), 3, 2),
    #t.gen_chunk("chunk test 1 - non-one output shape", torch.randint(0, 10, [5, 6, 5]), 2, 1),
    
    #t.gen_transpose("Transpose test 1 - 3 x 2", torch.randint(1, 10, [3, 2])),
    #t.gen_transpose("Transpose test 2 - transpose dims: 1 & 2 for matrix: 3 x 2 x 2", torch.randint(1, 10, [3, 2, 2]), 1, 2),
    #t.gen_transpose("Transpose test 3 - transpose dims: 0 & 2 for matrix: 3 x 2 x 2", torch.randint(1, 10, [3, 2, 2]), 0, 2),
    #t.gen_transpose("Transpose test 4 - transpose dims: 0 & 2 for matrix: 2 x 3 x 2", torch.randint(1, 10, [2, 3, 2]), 0, 2),
    #t.gen_transpose("Transpose test 5 - transpose dims: 0 & 2 for matrix: 2 x 2 x 3", torch.randint(1, 10, [2, 2, 3]), 0, 2),
    #t.gen_transpose("Transpose test 6 - transpose dims: 0 & 2 for matrix: 128 x 256", torch.randint(1, 10, [256, 128])),

    #t.gen_unsqueeze("Unsqueeze test 1", [10], 0),
    #t.gen_unsqueeze("Unsqueeze test 2", [10, 3, 2], 0),
    #t.gen_unsqueeze("Unsqueeze test 3", [10, 3, 2], 1),
    #t.gen_unsqueeze("Unsqueeze test 3", [10, 3, 2], -1),

    #t.gen_squeeze("Squeeze test 1", [10], 0),
    #t.gen_squeeze("Squeeze test 2", [10, 1, 2, 1], None),
    #t.gen_squeeze("Squeeze test 3", [10, 1, 2, 1], 1),
    #t.gen_squeeze("Squeeze test 3", [10, 1, 2, 1], -1),


    # Factory Tests -----------------------------------------------------------

    #t.gen_linspace("Linspace test 1", 1, 10, 10),
    #t.gen_linspace("Linspace test 2", 1, 10, 100),
    #t.gen_linspace("Linspace test 3", 0, 1, 50),
]
    
    

def build():
    out = json.dumps(tests)
    fp = open("./tests.json", "w")
    fp.write(out)
    fp.close()

if __name__ == "__main__":
    build()