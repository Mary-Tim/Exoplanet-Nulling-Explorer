import torch
from tensordict.prototype import tensorclass
from tensordict import TensorDict

#@tensorclass
#class Data:
#    a: torch.Tensor
#    b: torch.Tensor

data = TensorDict({
    "a": torch.randn(10),
    "b": torch.randn(10),
}, batch_size=[10])

def AplusB(data):
    #return data.a+data.b
    print(data['a'])
    return data['a']+data['b']

#data = Data(a=torch.randn(10), b=torch.randn(10), batch_size=[10])
print(data['a'])
result = torch.vmap(AplusB)(data)
print(result)