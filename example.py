import logging

from torch.utils.data import DataLoader

from src.transplanter import Transplanter
from test_utils.models import LinearNet
from test_utils.datasets import RandomDataset
from src.utilities.block_module import BlockModule
from src.utilities.logger import log_model_blocks

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    dataloader = DataLoader(dataset=RandomDataset(1000, (5), (5)),
                            batch_size=10,
                            shuffle=True)
    teacher_model = LinearNet(5, 5, 1)
    student_model = LinearNet(5, 5, 1)

    teacher_block_module = BlockModule(teacher_model)
    student_block_module = BlockModule(student_model)

    # log_model_blocks(teacher_block_module, student_block_module)

    # subnet = teacher_block_module.get_subnet(start_block=0,
    #                                          end_block=2)

    # print(subnet)

    # for var in vars(teacher_model):
    #     print(var)


    transplanter = Transplanter()
    # mapping = transplanter.map_blocks(teacher_block_module, student_block_module)
    # print(mapping)
    
    transplanter.transplant(teacher_model=teacher_model,
                            student_model=student_model,
                            dataloader=dataloader)


# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
# return hook