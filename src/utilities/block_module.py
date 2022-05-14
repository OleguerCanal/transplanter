"""Extension of torch.nn.Module allowing to operate
   on a 
"""

import torch

class BlockModule(torch.nn.Module):
    
    def __init__(self, model):
        super(BlockModule, self).__init__()
        self.model = model
        self.grouped_params = self._get_grouped_params()

    def __get_block_info(self, param_name : str) -> int:
        param_name_splitted = param_name.split(".")
        field = ""
        for pn in param_name_splitted:
            if pn.isnumeric():
                return int(pn), field[:-1]
            field += pn + "."
        return None, None

    def _get_grouped_params(self) -> list:
        """Get list of blocks composing the model
        """
        param_names = [name for name, _ in self.model.named_parameters()]
        grouped_param_names, current_group = [], []
        current_id = None
        for param_name in param_names:
            block_id, _ = self.__get_block_info(param_name)
            if block_id == current_id:
                current_group.append(param_name)
            else:
                if len(current_group) > 0:
                    grouped_param_names.append(current_group)
                current_group = [param_name]
                current_id = block_id
        grouped_param_names.append(current_group)
        return grouped_param_names
    
    def get_subnet(self,
                   start_block : int,
                   end_block : int):
        """Get the network from start_bloc (inclusive) to end_block (exclusive)
           TODO(Oleguer): This is suuper limited as for now...maybe use hooks?
        """
        start_block = 0 if start_block is None else start_block
        end_block = len(self) if end_block is None else end_block
        assert start_block >= 0 and start_block < len(self)
        assert end_block > 0 and end_block <= len(self)
        assert start_block < end_block

        start_block_index, start_field =\
            self.__get_block_info(self.grouped_params[start_block][0])
        end_block_index, end_field =\
            self.__get_block_info(self.grouped_params[end_block - 1][0])
        
        if start_field != end_field:
            raise NotImplementedError("Can only forward between blocks of the same module_list")
        if start_field is None:
            raise NotImplementedError("Can only forward blocks in module_lists")

        module_list = getattr(self.model, start_field)
        return module_list[start_block_index:end_block_index + 1]

    # def get_subnet(self, block_ids : list):
    #     module_list = getattr(self.model, start_field)
    #     return module_list[block_ids]

    def forward(self,
                start_block : int = None,
                end_block : int = None,
                **kwargs):
        """Forward model from start_block (inclusive) to end_block (not inclusive)
        """
        subnet = self.get_subnet(start_block=start_block,
                                 end_block=end_block)

        return subnet(kwargs)

    def __len__(self):
        return len(self.grouped_params)

    def __str__(self):
        s = ""
        for indx, block in enumerate(self.grouped_params):
            s += "Block " + str(indx) + ": " + str(block) + "\n"
        return s[:-1]