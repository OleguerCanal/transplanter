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

    # def get

    def transplant(self,
                   teacher_model : torch.nn.Module,
                   student_model : torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader) -> None:
        
        self.teacher_blocks = self.block_extractor.get_blocks(teacher_model)
        self.student_blocks = self.block_extractor.get_blocks(student_model)

        log_model_blocks(self.teacher_blocks, self.student_blocks)

        
        pass