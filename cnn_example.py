import logging
from torch.utils.data import DataLoader

from src.transplanter import Transplanter
from test_models.models import SmallConvNet, BigConvNet
from src.datasets import RandomDataset
from src.utilities.block_module import BlockModule
from src.utilities.helpers import InputShapes
from test_models.train_cnn import train

logging.basicConfig(level=logging.DEBUG)

def print_weights(model):
    state_dict = {p[0]: p[1] for p in model.named_parameters()}
    # print(state_dict["hidden_layers.1.conv.weight"].shape)
    # print(state_dict["hidden_layers.1.conv.bias"].shape)
    for item in state_dict:
        print(item, state_dict[item].shape)
    
    # print(state_dict["hidden_layers.0.conv.weight"][0])
    # print(state_dict["hidden_layers.0.conv.bias"][0])


if __name__ == "__main__":
    # Intermediate blocks shapes
    input_shapes_list = [
        InputShapes((3, 32, 32), (3, 32, 32)),
        InputShapes((64, 15, 15), (128, 15, 15)),
        InputShapes((64, 6, 6), (128, 6, 6)),
        InputShapes((1024), (2048)),
    ]

    teacher_args = {"hidden_dim": 64, "flattened_size": 1024}
    teacher_model = SmallConvNet.load_from_checkpoint(checkpoint_path="test_models/trained_models/test_1.ckpt", **teacher_args)
    student_model = BigConvNet(hidden_dim=128,
                               flattened_size=2048)

    print_weights(teacher_model)
    print_weights(student_model)
 
    teacher_block_module = BlockModule(teacher_model)
    student_block_module = BlockModule(student_model)

    # print("student_block_module")
    # print(student_block_module)

    transplanter = Transplanter()
    
    transplanter.transplant(teacher_model=teacher_model,
                            student_model=student_model,
                            input_shapes_list=input_shapes_list)

    print_weights(student_model)
    
    # student_model.out_layer_1.weight.data = teacher_model.out_layer_1.weight.data
    # # student_model.out_layer_1.bias.data = teacher_model.out_layer_1.bias.data
    # student_model.out_layer_2.weight.data = teacher_model.out_layer_2.weight.data
    # # student_model.out_layer_2.bias.data = teacher_model.out_layer_2.bias.data

    train(student_model, "student_0", data_dir="test_models/data")