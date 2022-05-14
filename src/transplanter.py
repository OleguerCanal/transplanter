import logging

import torch

from .utilities.block_module import BlockModule
from .utilities.logger import log_model_blocks
from .utilities.freezer import freeze
from .utilities.helpers import InputShapes, RandomInput, slice_tensor, overwrite_tensor, copy_weights, initialize_new_layer, same_layer_type

class Transplanter:

    def __init__(self) -> None:
        pass

    def map_blocks(self,
                    teacher_model : torch.nn.Module,
                    student_model : torch.nn.Module) -> None:
        len_teacher = len(teacher_model)
        len_student = len(student_model)
        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
        assignations = list(split(list(range(len_student)), len_teacher))
        mapping = {}
        for i, assigned in enumerate(assignations): 
            mapping[i] = assigned
        return mapping


              
    def transplant(self,
                   teacher_model : torch.nn.Module,
                   student_model : torch.nn.Module,
                   input_shapes_list: list) -> None:
        
        teacher_model = BlockModule(teacher_model)
        student_model = BlockModule(student_model)


        # Don't compute gradients on teacher model
        # freeze(teacher_model)
        # teacher_model.eval()
        block_mapping = self.map_blocks(teacher_model, student_model)
        print("block_mapping:", block_mapping)

        self._transfer_weights(block_mapping, teacher_model, student_model)