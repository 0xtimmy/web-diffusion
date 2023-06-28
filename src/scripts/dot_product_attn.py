import torch;
import torch.nn.functional as F
import math

def test_sdpa(q, k, v):
    return torch.softmax(q.mm(k) / math.sqrt(k.nelement()), 0).mm(v)

q = torch.rand(2, 64, 64)
k = torch.rand(2, 64, 64)
v = torch.rand(2, 64, 64)

actual = F.scaled_dot_product_attention(q, k, v)
test = torch.cat((test_sdpa(q[0].squeeze(0), k[0].squeeze(0), v[0].squeeze(0)).unsqueeze(0), test_sdpa(q[1].squeeze(0), k[1].squeeze(0), v[1].squeeze(0)).unsqueeze(0)), 0)

if(torch.equal(actual, test)): print("Equal!")
else: print("Not Equal!")

print("average difference: ", torch.mean(torch.abs(actual - test)))
print("variance of difference: ", torch.var(actual - test))