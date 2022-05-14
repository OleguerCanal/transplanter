import copy
import logging

import torch
from .utilities.block_module import BlockModule
from .utilities.helpers import overwrite_tensor

class WeightCopier:
    def __init__(self):
        pass

    def copy(self,
             mapping : dict,
             teacher_model : BlockModule,
             student_model : BlockModule,
             name_adaptor : callable = None):
        logging.info("Copying weights...")
        teacher_state_dict = teacher_model.state_dict()
        student_state_dict = student_model.state_dict()
        for i, assigned in mapping.items():
            t_params = [name_adaptor(p) for p in teacher_model.grouped_params[i]]
            s_params = [name_adaptor(p) for a in assigned for p in student_model.grouped_params[a]]
            for t_param_name, s_param_name in zip(t_params, s_params[:len(t_params)]):
                assert self._same_layer_type(t_param_name, s_param_name)
                s_param = student_state_dict[s_param_name]
                t_param = teacher_state_dict[t_param_name]
                student_state_dict[s_param_name] = self._copy_weights(teacher_weights=t_param,
                                                                student_weights=s_param)

            if len(s_params) > len(t_params):
                for layer_name in s_params[len(t_params):]:
                    student_state_dict = self._initialize_new_layer(layer_name=layer_name,
                                                                    state_dict=student_state_dict)
        student_model.load_state_dict(student_state_dict)
        logging.info("Copying weights...DONE")


    def _copy_weights(self,
                     teacher_weights: torch.Tensor,
                     student_weights: torch.Tensor,
                     initial_weights_std: float = 0.01):
        student_weights = torch.rand_like(student_weights)*initial_weights_std
        student_weights = overwrite_tensor(big_tensor=student_weights,
                                           small_tensor=teacher_weights)
        return student_weights

    def _initialize_new_layer(self, layer_name: str, state_dict: dict):
        if "conv" in layer_name:
            if "weight" in layer_name:
                state_dict[layer_name] = self._new_conv_weight(
                    state_dict[layer_name])
            elif "bias" in layer_name:
                state_dict[layer_name] = self._new_conv_bias(state_dict[layer_name])
            else:
                raise Exception(
                    "Unknown conv layer type: neither weight nor bias")
        else:
            if "weight" in layer_name:
                state_dict[layer_name] = self._new_linear_weight(
                    state_dict[layer_name])
            elif "bias" in layer_name:
                state_dict[layer_name] = self._new_linear_bias(
                    state_dict[layer_name])
            else:
                raise Exception(
                    "Unknown linear layer type: neither weight nor bias")
        return state_dict

    def _new_linear_weight(self,
                           layer_weights: torch.Tensor,
                           initial_weights_std: float = 0.01):
        layer_weights = torch.rand_like(layer_weights)*initial_weights_std
        for i in range(min(layer_weights.shape)):
            layer_weights[i, i] = 1
        return layer_weights

    def _new_linear_bias(self,
                         layer_bias: torch.Tensor,
                         initial_weights_std: float = 0.01):
        return torch.rand_like(layer_bias)*initial_weights_std

    def _new_conv_weight(self,
                         layer_weights: torch.Tensor,
                         initial_weights_std: float = 0.01):
        layer_weights = torch.rand_like(layer_weights)*initial_weights_std
        n_filters = layer_weights.shape[0]
        input_channels = layer_weights.shape[1]
        for i in range(min(n_filters, input_channels)):
            layer_weights[i, i, 0, 0] = 1
        return layer_weights

    def _new_conv_bias(self,
                       layer_bias: torch.Tensor,
                       initial_weights_std: float = 0.01):
        return torch.rand_like(layer_bias)*initial_weights_std

    def _same_layer_type(self, layer_1: str, layer_2: str):
        if "conv" in layer_1 and not "conv" in layer_2:
            return False
        if not "conv" in layer_1 and "conv" in layer_2:
            return False
        if ("weight" in layer_1 and "weight" in layer_2) or\
                ("bias" in layer_1 and "bias" in layer_2):
            return True
        return False
