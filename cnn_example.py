import logging

from torch.utils.data import DataLoader

from src.transplanter import Transplanter
from test_models.models import ConvNet
from src.datasets import RandomDataset
from src.utilities.block_module import BlockModule
from test_models.train_cnn import train

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Intermediate bloks shapes
    shapes = [(3, 32, 32), (64, 15, 15), (64, 6, 6), (1024)]

    dataloaders = [
        DataLoader(dataset=RandomDataset(1000, shapes[i]), batch_size=10, shuffle=True)
        for i in range(len(shapes))
    ]

    teacher_args = {"in_features": 3, "hidden_dim": 64, "out_features": 10, "n_blocks": 3}
    teacher_model = ConvNet.load_from_checkpoint(checkpoint_path="test_models/trained_models/test_1.ckpt", **teacher_args)
    student_model = ConvNet(3, hidden_dim=64, out_features=10, n_blocks=3)

    teacher_block_module = BlockModule(teacher_model)
    student_block_module = BlockModule(student_model)

    print("student_block_module")
    print(student_block_module)

    transplanter = Transplanter()
    mapping = transplanter.map_blocks(teacher_block_module, student_block_module)
    print("mapping")
    print(mapping)
    
    # transplanter.transplant(teacher_model=teacher_model,
    #                         student_model=student_model,
    #                         dataloaders=dataloaders)

    teacher_state_dict = {p[0]: p[1] for p in teacher_model.named_parameters()}
    print(teacher_state_dict["hidden_layers.0.conv.weight"].shape)
    print(teacher_state_dict["hidden_layers.0.conv.bias"].shape)
    # student_model.load_state_dict(teacher_state_dict)
    # student_model.out_layer_1.weight.data = teacher_model.out_layer_1.weight.data
    # student_model.out_layer_1.bias.data = teacher_model.out_layer_1.bias.data
    # student_model.out_layer_2.weight.data = teacher_model.out_layer_2.weight.data
    # student_model.out_layer_2.bias.data = teacher_model.out_layer_2.bias.data


    # train(student_model, "student_0", data_dir="test_models/data")