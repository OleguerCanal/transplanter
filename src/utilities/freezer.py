import logging

def freeze(model, str_in_layers=[""]):
    """Freeze all layers containing str_in_layers in the name
       E.g. 
        freeze(model, str_in_layers=["encoder"]) to freeze all encoder layers
        freeze(model, str_in_layers=[""]) to freeze the whole model
    """
    assert type(str_in_layers) == list
    for name, param in model.named_parameters():
        if any(substring in name for substring in str_in_layers):
            logging.info("Not requiring grad on: " + name)
            param.requires_grad = False
    return model