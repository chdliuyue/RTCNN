import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import norm
from models_pytorch.utils import create_dataset, ce_loss, DS_Combin
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有可用的GPU，使用GPU
else:
    device = torch.device("cpu")  # 如果没有可用的GPU，使用CPU


def get_inverse_Hessian(model, X_train, Q_train, labels, layer_name='Utilities'):
    data_size = X_train.shape[0]
    NUM_CHOICES = X_train.shape[2]
    dataset = create_dataset(X_train, Q_train, labels)
    loader = DataLoader(dataset, batch_size=X_train.shape[0], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(loader):
        XX = data[0].to(device)
        QQ = data[1].to(device)
        yy = data[2].to(device)
        model.zero_grad()
        evidence = model(XX, QQ)
        loss = 0
        alpha = dict()
        for v_num in range(len(evidence)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(yy, alpha[v_num], NUM_CHOICES, 200, 100)
        alpha_a = DS_Combin(alpha, NUM_CHOICES)
        loss += ce_loss(yy, alpha_a, NUM_CHOICES, 200, 100)
        loss = torch.mean(loss)

    # Get layer and gradient w.r.t. loss
    beta_layer = model._modules[layer_name]
    beta_gradient = torch.autograd.grad(loss, beta_layer.parameters(), create_graph=True)

    # Get second order derivative operators (linewise of Hessian)
    beta_layer_weight = next(beta_layer.parameters())
    Hessian_lines_op = []
    for i in range(len(beta_gradient[0].view(-1))):
        Hessian_lines_op.append(torch.autograd.grad(beta_gradient[0].view(-1)[i], beta_layer_weight, create_graph=True))

    # Line by line Hessian average multiplied by data length (due to automatic normalization)
    Hessian = [torch.cat([line_op[i].flatten() for i in range(len(line_op))]) for line_op in Hessian_lines_op]
    Hessian = torch.stack(Hessian) * data_size
    # Hessian = torch.stack(Hessian)

    # The inverse Hessian:
    try:
        invHess = torch.linalg.inv(Hessian)
    except torch.linalg.LinAlgError:
        print('Singular matrix')
        return np.nan

    return invHess.cpu().detach().numpy()


def get_stds(model, X_train, Q_train, labels, layer_name='Utilities'):
    inv_Hess = get_inverse_Hessian(model, X_train, Q_train, labels, layer_name)

    if isinstance(inv_Hess, float):
        return np.nan
    else:
        stds = [inv_Hess[i, i] ** 0.5 for i in range(inv_Hess.shape[0])]
        return np.array(stds).flatten()


def model_summary(trained_model, X_train, Q_train, y_train, X_vars_names=[], Q_vars_names=[]):
    emb_betas_stds = get_stds(trained_model, X_train, Q_train, y_train, layer_name='utilities1')
    exog_betas_stds = get_stds(trained_model, X_train, Q_train, y_train, layer_name='utilities2')

    betas_embs = trained_model.utilities1.weight.detach().cpu().numpy().flatten()
    betas_exog = trained_model.utilities2.weight.detach().cpu().numpy().flatten()

    if not isinstance(emb_betas_stds, float) and not isinstance(exog_betas_stds, float):
        z_embs = betas_embs / emb_betas_stds
        p_embs = (1 - norm.cdf(np.abs(z_embs))) * 2
        z_exog = betas_exog / exog_betas_stds
        p_exog = (1 - norm.cdf(np.abs(z_exog))) * 2

        stats_exog = np.array(list(zip(X_vars_names, betas_exog, exog_betas_stds, z_exog, p_exog)))
        stats_embs = np.array(list(zip(Q_vars_names, betas_embs, emb_betas_stds, z_embs, p_embs)))
        stats_all = np.vstack([stats_exog, stats_embs])

        df_stats = pd.DataFrame(index=[i[0] for i in stats_all],
                                data=np.array([[float(i[1]) for i in stats_all],
                                               [float(i[2]) for i in stats_all],
                                               [float(i[3]) for i in stats_all],
                                               [np.round(float(i[4]), 4) for i in stats_all]]).T,
                                columns=['Betas', 'St errors', 't-stat', 'p-value'])

        return df_stats
    else:
        return np.nan


def create_index(alfabet): # alphabet-->number of unique categories
    """ Maps categories (strings) to integers and creates
        a dictionary with look up index. """
    index2alfa = {}
    alfa2index = {}

    for i in range(len(alfabet)):
        index2alfa[i] = alfabet[i]
        alfa2index[alfabet[i]] = i
    return index2alfa, alfa2index


def get_betas_and_embeddings(trained_model, Q_df_train):
    UNIQUE_CATS = sorted(list(set(Q_df_train.values.reshape(-1))))
    DICT = {}
    DICT['index2alfa_from'], DICT['alfa2index_from'] = create_index(UNIQUE_CATS)
    DICT['index2alfa_from'], DICT['alfa2index_from'] = create_index(UNIQUE_CATS)
    betas_embs = next(trained_model._modules['utilities1'].parameters()).reshape(-1)
    betas_exog = next(trained_model._modules['utilities2'].parameters()).reshape(-1)
    embeddings = next(trained_model._modules['embeddings'].parameters())

    DICT['embeddings'] = embeddings
    DICT['betas_embs'] = betas_embs
    DICT['betas_exog'] = betas_exog

    return DICT

