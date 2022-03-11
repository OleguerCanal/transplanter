import torch

class BlockExtractor:

    def __init__(self) -> None:
        pass

    def _get_block_id(self, param_name : str):
        param_name_splitted = param_name.split(".")
        for pn in param_name_splitted:
            if pn.isnumeric():
                return int(pn)
        return None

    def get_blocks(self,
                   model : torch.nn.Module):
        param_names = [name for name, _ in model.named_parameters()]
        
        grouped_param_names = []
        current_group = []
        current_id = None
        for param_name in param_names:
            block_id = self._get_block_id(param_name)
            if block_id == current_id:
                current_group.append(param_name)
            else:
                if len(current_group) > 0:
                    grouped_param_names.append(current_group)
                current_group = [param_name]
                current_id = block_id
        grouped_param_names.append(current_group)
        return grouped_param_names