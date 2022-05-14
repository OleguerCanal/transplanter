import logging

import torch
import torch.optim as optim
from tqdm import tqdm

from .utilities.block_module import BlockModule
from .utilities.helpers import InputShapes, slice_tensor, RandomInput, overwrite_tensor

class BlockTrainer:
    def __init__(self):
        pass

    def finetune(self,
                 block_mapping : dict,
                 input_shapes_list : list,
                 teacher_model : BlockModule, 
                 student_model : BlockModule):
        for i, assigned in block_mapping.items():
            logging.info("Finetuning... %i %s"%(i, assigned))
            try:
                teacher_subnet = teacher_model.get_subnet(i, i+1)
                student_subnet = student_model.get_subnet(assigned[0], assigned[-1] + 1)
            except NotImplementedError as e:
                logging.error(e)
                continue
            self._finetune_block(
                teacher=teacher_subnet,
                student=student_subnet,
                input_shapes=input_shapes_list[i],
            )
        logging.info("Finetuning done.")

    def _finetune_block(self,
                        teacher : torch.nn.Module,
                        student : torch.nn.Module,
                        input_shapes : InputShapes):
        optimizer = optim.Adam(student.parameters(), lr=1e-4)

        pbar = tqdm(list(range(1000)))
        for _ in pbar:
            teacher_input, student_input = self._get_inputs(input_shapes, batch_size=10)
            optimizer.zero_grad()
            teacher_output = self._forward(teacher, teacher_input)
            student_output = self._forward(student, student_input)
            loss = self.__loss(student_output, teacher_output)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    def _forward(self,
                  module_list : torch.nn.ModuleList,
                  x : torch.Tensor) -> tuple:
        for module in module_list:
            x = module(x)
        return x

    def __loss(self,
            guess : torch.Tensor,
            target : torch.Tensor) -> torch.Tensor:
        guess = slice_tensor(guess, target.shape)
        dist_loss = torch.nn.MSELoss()(guess, target)
        return dist_loss

    def _get_inputs(self,
                    input_shapes : InputShapes,
                    batch_size : int) -> tuple:
        teacher_input = RandomInput(input_shapes.teacher).get_batch(batch_size)
        student_input = RandomInput(input_shapes.student).get_batch(batch_size)
        student_input = overwrite_tensor(student_input, teacher_input)
        return teacher_input, student_input


