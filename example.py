import logging

from torch.utils.data import DataLoader

from src.transplanter import Transplanter
from test_utils.models import Net
from test_utils.datasets import RandomDataset

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    dataloader = DataLoader(dataset=RandomDataset(100, (10), (10)),
                            batch_size=10,
                            shuffle=True)
    teacher_model = Net(5, 10, 3)
    student_model = Net(10, 10, 4)

    transplanter = Transplanter(batch_size=10)
    transplanter.transplant(teacher_model=teacher_model,
                            student_model=student_model,
                            dataloader=dataloader)


# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
# return hook