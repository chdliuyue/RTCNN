import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, r2_score


if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有可用的GPU，使用GPU
else:
    device = torch.device("cpu")  # 如果没有可用的GPU，使用CPU


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(label, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    # label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return A + B


def DS_Combin(alpha, choices_num):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """

    def DS_Combin_two(alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = choices_num / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, choices_num, 1), b[1].view(-1, 1, choices_num))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = choices_num / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
    return alpha_a


# 将离散数据依次映射成数字
def cats2ints(Q_df):
    UNIQUE_CATS = sorted(list(set(Q_df.values.reshape(-1))))
    cat2index = {}
    for i in range(len(UNIQUE_CATS)):
        cat2index[UNIQUE_CATS[i]] = i

    return cat2index


# 将string转成int
def cats2ints_transform(Q_df, cat2index):
    Q_int = []
    for obs in Q_df.values:
        input_i = [cat2index[cat] for cat in obs]
        Q_int.append(input_i)

    return np.array(Q_int)


# 将X, Q, Y放在一起实现dataloader可以迭代
def create_dataset(x_data, q_data, y_data):
    class MyDataset(Dataset):
        def __init__(self):
            self.x_data = x_data
            self.q_data = q_data
            self.y_data = y_data

        def __len__(self):
            # 返回数据数量
            return len(self.x_data)

        def __getitem__(self, idx):
            # 返回一个数据样本和对应的索引
            return self.x_data[idx], self.q_data[idx], self.y_data[idx]

    return MyDataset()


# 计算多分类问题的准确率（ACC）
def calculate_accuracy(preds, targets):
    """
    参数：preds和targets的形状为 [batch_size, num_classes] 的概率分布张量
    返回：accuracy：准确率，一个浮点数表示预测正确的样本所占比例
    """
    preds_classes = torch.argmax(preds, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    correct = (preds_classes == true_classes).sum().item()
    accuracy = correct / targets.shape[0]

    return accuracy


# 计算似然对数（LL）
def calculate_ll(preds, targets):
    """
    参数：preds和targets的形状为 [batch_size, num_classes] 的概率分布张量
    返回：LL：似然对数，(-, 0]
    """
    preds = F.softmax(preds, dim=1)
    # preds = F.log_softmax(preds, dim=1)
    preds = preds.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    # ll = sum([np.log(x) for x in np.multiply(preds.reshape(-1), targets.reshape(-1)) if x != 0.0])
    # ll = - np.sum(targets * np.log(preds)) / 7234
    ll = np.sum(targets * np.log(preds))

    return ll


def calculate_f1(preds, targets):
    max_preds = torch.argmax(preds, dim=1).cpu().numpy()
    max_targets = torch.argmax(targets, dim=1).cpu().numpy()
    f1 = f1_score(max_preds, max_targets, average='macro')

    return f1


def t_model_predict(X, Q, y, model, epoch, lambda_epochs):
    NUM_CHOICES = X.shape[2]
    dataset = create_dataset(X, Q, y)
    loader = DataLoader(dataset, batch_size=X.shape[0], shuffle=True)
    for i, data in enumerate(loader):
        XX = data[0].to(device)
        QQ = data[1].to(device)
        yy = data[2].to(device)
        evidence = model(XX, QQ)
        loss = 0
        alpha = dict()
        s = dict()
        for v_num in range(len(evidence)):
            alpha[v_num] = evidence[v_num] + 1
            s[v_num] = torch.sum(alpha[v_num], dim=1)
            loss += ce_loss(yy, alpha[v_num], NUM_CHOICES, epoch, lambda_epochs)
        alpha_a = DS_Combin(alpha, NUM_CHOICES)
        evidence_a = alpha_a - 1
        loss += ce_loss(yy, alpha_a, NUM_CHOICES, epoch, lambda_epochs)
        loss = torch.mean(loss).item()
        acc = calculate_accuracy(evidence_a, yy)
        f1 = calculate_f1(evidence_a, yy)

    return loss, acc, f1

