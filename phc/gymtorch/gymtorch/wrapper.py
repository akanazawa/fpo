from isaacgym import gymapi
import torch
import gymtorch._C as gt  # This will import the compiled C++ module


def create_context(device="cuda:0"):
    """Force PyTorch to create a primary CUDA context on the specified device"""
    torch.zeros([1], device=device)


def wrap_tensor(gym_tensor, offsets=None, counts=None):
    data = gym_tensor.data_ptr
    device = gym_tensor.device
    dtype = int(gym_tensor.dtype)
    shape = gym_tensor.shape
    if offsets is None:
        offsets = tuple([0] * len(shape))
    if counts is None:
        counts = shape
    return gt.wrap_tensor_impl(data, device, dtype, shape, offsets, counts)


def torch2gym_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return gymapi.DTYPE_FLOAT32
    elif torch_dtype == torch.uint8:
        return gymapi.DTYPE_UINT8
    elif torch_dtype == torch.int16:
        return gymapi.DTYPE_INT16
    elif torch_dtype == torch.int32:
        return gymapi.DTYPE_UINT32
    elif torch_dtype == torch.int64:
        return gymapi.DTYPE_UINT64
    else:
        raise Exception("Unsupported Gym tensor dtype")


def torch2gym_device(torch_device):
    if torch_device.type == "cpu":
        return -1
    elif torch_device.type == "cuda":
        return torch_device.index
    else:
        raise Exception("Unsupported Gym tensor device")


def unwrap_tensor(torch_tensor):
    if not torch_tensor.is_contiguous():
        raise Exception("Input tensor must be contiguous")
    gym_tensor = gymapi.Tensor()
    gym_tensor.device = torch2gym_device(torch_tensor.device)
    gym_tensor.dtype = torch2gym_dtype(torch_tensor.dtype)
    gym_tensor.shape = list(torch_tensor.shape)
    gym_tensor.data_address = torch_tensor.data_ptr()
    gym_tensor.own_data = False
    return gym_tensor
