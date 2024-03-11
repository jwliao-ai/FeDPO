import copy
import numpy as np
import torch
import sys
import os
from collections import OrderedDict


def agg_FedAvg(state_dicts):
    '''
    FedAvg aggregation
    param w_locals: list of (model_state_dicts)
    return a state_dict
    '''

    aggregated_state_dict = OrderedDict()

    for state_dict in state_dicts:
        for key, value in state_dict.items():
            if key not in aggregated_state_dict:
                aggregated_state_dict[key] = value
            else:
                aggregated_state_dict[key] = aggregated_state_dict[key] + value

    num_state_dicts = len(state_dicts)
    for key, value in aggregated_state_dict.items():
        aggregated_state_dict[key] = value / num_state_dicts

    return aggregated_state_dict


def compute_similarity(args, s_locals):
    '''
    compute the data distribution similarity of clients
    param s_locals: dict{layer_name: data distribution representation}
    return: list(client_num * client_num * similarity)
    '''
    client_num = len(s_locals)
    # similarities_dict = [[] for _ in range(client_num)] # dict{layer_name: similarity}
    similarities = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            # similarities_dict[i].append(dict())
            for k in s_locals[i]:
                if args.similarity_method == 0:
                    similarities[i][j] += torch.cosine_similarity(
                        s_locals[i][k], s_locals[j][k], dim=0).item()
                elif args.similarity_method == 1:
                    similarities[i][j] += 1.0 / torch.dist(
                        s_locals[i][k], s_locals[j][k], p=1).item()
                elif args.similarity_method == 2:
                    similarities[i][j] += 1.0 / torch.dist(
                        s_locals[i][k], s_locals[j][k], p=2).item()
                else:
                    assert False

    return similarities
