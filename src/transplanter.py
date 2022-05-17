import logging

import torch

from .block_trainer import BlockTrainer
from .weight_copier import WeightCopier
from .utilities.block_mapper import map_blocks
from .utilities.block_module import BlockModule

class Transplanter:

    def __init__(self) -> None:
        pass
              
    def transplant(self,
                   teacher_model : torch.nn.Module,
                   student_model : torch.nn.Module,
                   input_shapes_list: list,
                   copy_weights : bool = True,
                   finetune_blocks : bool = True,) -> None:
        
        teacher_model = BlockModule(teacher_model)
        student_model = BlockModule(student_model)

        
        block_mapping = map_blocks(teacher_model=teacher_model,
                                   student_model=student_model,
                                   show=True)
        if copy_weights:
            layer_name_adaptor = lambda x : "model." + x
            wc = WeightCopier()
            wc.copy(mapping=block_mapping,
                    teacher_model=teacher_model,
                    student_model=student_model,
                    name_adaptor=layer_name_adaptor)
        if finetune_blocks:
            bt = BlockTrainer()
            bt.finetune(mapping=block_mapping,
                        input_shapes_list=input_shapes_list,
                        teacher_model=teacher_model,
                        student_model=student_model)