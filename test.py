import torch

print(torch.tensor(None, dtype=torch.long))

# sparse_tensor = torch.sparse_coo_tensor((range(10), range(10)), [1] * 10, size=(10, 10), dtype=torch.float)
# print(sparse_tensor)

# sampled = sparse_tensor[[1, 2, 3]]
# print(sampled)

# sparse_tensor_gpu = sparse_tensor.to('cuda:0')
# print(sparse_tensor_gpu)

# sampled = sparse_tensor[[1, 2, 3]]
# print(sampled)