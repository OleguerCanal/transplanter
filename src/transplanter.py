import logging

import torch
import torch.optim as optim
from tqdm import tqdm

from src.utilities.block_extractor import BlockExtractor
from src.utilities.block_module import BlockModule
from src.utilities.logger import log_model_blocks
from src.utilities.freezer import freeze

class Transplanter:

    def __init__(self) -> None:
        self.block_extractor = BlockExtractor()

    def __loss(self,
               input : torch.Tensor,
               target : torch.Tensor) -> torch.Tensor:
        dist_loss = -(input*target).sum(dim=-1).mean()
        return dist_loss

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

    def __forward(self, module_list : torch.nn.ModuleList, x : torch.Tensor) -> tuple:
        for module in module_list:
            x = module(x)
        return x

    def _finetune_block(self,
                        teacher : torch.nn.Module,
                        student : torch.nn.Module,
                        dataloader : torch.utils.data.DataLoader):
        optimizer = optim.Adam(student.parameters(), lr=0.0001)
        for _ in range(10):
            for batch in tqdm(dataloader):
                inputs, outputs = batch
                optimizer.zero_grad()
                teacher_output = self.__forward(teacher, inputs)
                student_output = self.__forward(student, inputs)
                loss = self.__loss(student_output, teacher_output)
                loss.backward()
                optimizer.step()
            print(loss)

    def transplant(self,
                   teacher_model : torch.nn.Module,
                   student_model : torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader) -> None:
        
        teacher_model = BlockModule(teacher_model)
        student_model = BlockModule(student_model)

        # Don't compute gradients on teacher model
        freeze(teacher_model)
        block_mapping = self.map_blocks(teacher_model, student_model)
        for i, assigned in block_mapping.items():
            logging.info("Finetuning... " + i + " " + assigned)
            try:
                teacher_subnet = teacher_model.get_subnet(i, i+1)
                student_subnet = student_model.get_subnet(assigned[0], assigned[-1] + 1)
            except NotImplementedError as e:
                logging.error(e)
                continue
            self._finetune_block(
                teacher=teacher_subnet,
                student=student_subnet,
                dataloader=dataloader,
            )
        logging.info("Finetuning done.")

