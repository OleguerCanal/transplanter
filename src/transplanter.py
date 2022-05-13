import logging

import torch
import torch.optim as optim
from tqdm import tqdm

from .utilities.block_extractor import BlockExtractor
from .utilities.block_module import BlockModule
from .utilities.logger import log_model_blocks
from .utilities.freezer import freeze
from .utilities.helpers import InputShapes, RandomInput, slice_tensor, overwrite_tensor, copy_weights, initialize_new_layer, same_layer_type

class Transplanter:

    def __init__(self) -> None:
        self.block_extractor = BlockExtractor()

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


    def _finetune_block(self,
                        teacher : torch.nn.Module,
                        student : torch.nn.Module,
                        input_shapes : InputShapes):
        optimizer = optim.Adam(student.parameters(), lr=1e-4)

        pbar = tqdm(list(range(1000)))
        for _ in pbar:
            teacher_input, student_input = self._get_inputs(input_shapes, batch_size=10)
            optimizer.zero_grad()
            teacher_output = self.__forward(teacher, teacher_input)
            student_output = self.__forward(student, student_input)
            loss = self.__loss(student_output, teacher_output)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    def _adapt_name(self, name : str) -> str:
        return "model." + name

    def _transfer_weights(self,
                          mapping : dict,
                          teacher_model : BlockModule,
                          student_model : BlockModule):
        teacher_state_dict = teacher_model.state_dict()
        student_state_dict = student_model.state_dict()
        for i, assigned in mapping.items():
            t_params = [self._adapt_name(p) for p in teacher_model.grouped_params[i]]
            s_params = [self._adapt_name(p) for a in assigned for p in student_model.grouped_params[a]]
            for t_param_name, s_param_name in zip(t_params, s_params[:len(t_params)]):
                assert same_layer_type(t_param_name, s_param_name)
                # NOTE: adapt_name is doing nothing
                s_param = student_state_dict[s_param_name]
                t_param = teacher_state_dict[t_param_name]
                student_state_dict[s_param_name] = copy_weights(teacher_weights=t_param,
                                                                student_weights=s_param)

            if len(s_params) > len(t_params):
                for layer_name in s_params[len(t_params):]:
                    student_state_dict = initialize_new_layer(layer_name=layer_name,
                                                              state_dict=student_state_dict)

        student_model.load_state_dict(student_state_dict)
                
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

# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
# return hook