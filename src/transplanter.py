import torch

from src.utilities.block_extractor import BlockExtractor
from src.utilities.logger import log_model_blocks

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

    def transplant(self,
                   teacher_model : torch.nn.Module,
                   student_model : torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader) -> None:
        pass