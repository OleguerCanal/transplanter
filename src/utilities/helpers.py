from collections import namedtuple
import torch
import copy

InputShapes = namedtuple('InputShapes', 'teacher student')

class RandomInput:
    def __init__(self, shape: tuple):
        self.shape = shape

    def get_batch(self, batch_size : int):
        return torch.rand((batch_size,) + self.shape)

def slice_tensor(tensor: torch.Tensor, shape : tuple):
    s = shape
    if len(s) == 1:
        return tensor[0:s[0]]
    if len(s) == 2:
        return tensor[0:s[0], 0:s[1]]
    if len(s) == 3:
        return tensor[0:s[0], 0:s[1], 0:s[2]]
    if len(s) == 4:
        return tensor[0:s[0], 0:s[1], 0:s[2], 0:s[3]]
    raise Exception('Tensor shape not supported')

def overwrite_tensor(big_tensor : torch.Tensor,
                     small_tensor : torch.Tensor):
    s = small_tensor.shape
    if len(s) == 1:
        big_tensor[0:s[0]] = copy.deepcopy(small_tensor)
    elif len(s) == 2:
        big_tensor[0:s[0], 0:s[1]] = copy.deepcopy(small_tensor)
    elif len(s) == 3:
        big_tensor[0:s[0], 0:s[1], 0:s[2]] = copy.deepcopy(small_tensor)
    elif len(s) == 4:
        big_tensor[0:s[0], 0:s[1], 0:s[2], 0:s[3]] = copy.deepcopy(small_tensor)
    elif len(s) == 5:
        big_tensor[0:s[0], 0:s[1], 0:s[2], 0:s[3], 0:s[4]] = copy.deepcopy(small_tensor)
    return big_tensor

def copy_weights(teacher_weights : torch.Tensor,
                 student_weights : torch.Tensor,
                 initial_weights_std : float = 0.01):
    student_weights = torch.rand_like(student_weights)*initial_weights_std
    student_weights = overwrite_tensor(big_tensor=student_weights,
                                       small_tensor=teacher_weights)
    return student_weights

def initialize_new_layer(layer_name : str, state_dict : dict):
    if "conv" in layer_name:
        if "weight" in layer_name:
            state_dict[layer_name] = new_conv_weight(state_dict[layer_name])
        elif "bias" in layer_name:
            state_dict[layer_name] = new_conv_bias(state_dict[layer_name])
        else:
            raise Exception("Unknown conv layer type: neither weight nor bias")
    elif "linear" in layer_name:
        if "weight" in layer_name:
            state_dict[layer_name] = new_linear_weight(state_dict[layer_name])
        elif "bias" in layer_name:
            state_dict[layer_name] = new_linear_bias(state_dict[layer_name])
        else:
            raise Exception("Unknown linear layer type: neither weight nor bias")
    else:
        raise Exception("Unknown layer type: neither conv nor linear")
    return state_dict



def new_linear_weight(layer_weights : torch.Tensor):
    pass

def new_linear_bias(layer_bias : torch.Tensor):
    pass

def new_conv_weight(layer_weights : torch.Tensor):
    pass

def new_conv_bias(layer_bias : torch.Tensor):
    pass