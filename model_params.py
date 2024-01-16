import copy


def get_model_params(model):
    return copy.deepcopy(model.cpu().state_dict())


def set_model_params(model, model_parameters):
    model.load_state_dict(copy.deepcopy(model_parameters))
