import copy
import numpy as np
import torch
import sys
import os


def agg_FedAvg(w_locals):
    '''
    FedAvg aggregation
    param w_locals: list of (sample_num, model_params)
    '''
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num

    (sample_num, averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params


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
