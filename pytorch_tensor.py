import torch


def tensor_creation():
    """example of tensor creation"""
    zeros = torch.zeros((3, 4))
    print(f"全零张量：\n{zeros}\ntype: {zeros.dtype}")

    ones = torch.ones((2, 3))
    print(f"全一张量：\n{ones}\ntype: {ones.dtype}")

    random_tensor = torch.rand((2, 3))
    print(f"随机张量：\n{random_tensor}\ntype: {random_tensor.dtype}")

    list_tensor = torch.tensor([[i for i in range(1, 4)], [i for i in range(4, 7)]])
    print(f"从列表创建的张量：\n{list_tensor}\ndevice: {list_tensor.device}")

    arange_tensor = torch.arange(1, 13).reshape((3, 4))
    print(f"arange tensor:\n{arange_tensor}\ntype: {arange_tensor.dtype}")


def tensor_operation():
    """"example of tensor operation"""
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    print(f"a =\n{a}\nb =\n{b}\n")

    # 逐元素运算
    print(f"a + b =\n{a + b}")
    print(f"a - b =\n{a - b}")
    print(f"a * b =\n{a * b}")
    print(f"a / b =\n{a / b}")
    print(f"b // a =\n{b // a}")

    # 矩阵运算
    print(f"a @ b =\n{a @ b}")

    # 其他数学运算
    c = a.to(torch.float)
    print(f"c =\n{c}")
    print(f"sqrt(c) =\n{torch.sqrt(c)}")
    print(f"exp(c) =\n{torch.exp(c)}")
    print(f"log(c) =\n{torch.log(c)}")
    print(f"sum(c) =\n{torch.sum(c, dim=-1, keepdim=True)}")
    print(f"mean(c) =\n{torch.mean(c, dim=-1, keepdim=True)}")


def tensor_shaping():
    """"tensor shaping"""
    x = torch.arange(12)
    print(f"shape of x: {x.shape}")

    x_viewed = x.view(3, 4)
    print(f"x_viewed =\n{x_viewed}")
    x[0] = 999
    print(f"x_viewed(after x changed) =\n{x_viewed}")
    x[0] = 0

    x_reshaped = x.reshape(2, 6)
    print(f"x_reshaped =\n{x_reshaped}")
    x[1] = 999
    print(f"x_reshaped(after x changed) =\n{x_reshaped}")
    x[1] = 1

    x = torch.arange(15).reshape(3, 5)
    print(f"shape of x: {x.shape}")
    print(f"transposed x:\n{x.t()}")

    print(f"x.unsqueeze(0) =\n{x.unsqueeze(0)}")
    print(f"x.unsqueeze(1) =\n{x.unsqueeze(1)}")
    print(f"x.unsqueeze(2) =\n{x.unsqueeze(2)}")

    a = torch.ones(2, 3)
    b = torch.zeros(2, 3)
    print(f"torch.cat((a, b)) =\n{torch.cat((a, b, torch.ones(1, 3)))}")


def tensor_indexing():
    """tensor indexing"""
    x = torch.arange(15).reshape(3, 5)
    print(f"x =\n{x}")
    print(f"x[2, 2] = {x[2, 2].item()}")
    print(f"x[1, ] = {x[1, :]}")
    print(f"x[:, 1] = {x[:, 1]}")
    print(f"x[1:3, 1:3] =\n{x[1:3, 1:3]}")

    # 布尔索引
    mask = x > 2
    print(f"x > 2 =\n{mask}")
    print(f"x[mask] =\n{x[mask]}")


if __name__ == "__main__":
    # tensor_creation()
    # tensor_operation()
    # tensor_shaping()
    tensor_indexing()
