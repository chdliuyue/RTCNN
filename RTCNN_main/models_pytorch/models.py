import torch
import torch.nn as nn
import torch.nn.functional as F


# Keras input (batch_size, height, width, channels)
# pytorch input (batch_size, channels, height, width)
def MNL(vars_num, choices_num, logits_activation='softmax'):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=(vars_num, 1), stride=1, padding=0, bias=False)
            self.activation = F.softmax if logits_activation == 'softmax' else F.relu

        def forward(self, x):
            x = self.conv(x)
            x = x.reshape(-1, choices_num)
            x = self.activation(x, dim=1)
            return x

    model = Model()
    print(model)

    return model


def TE_MNL(cont_vars_num, emb_vars_num, choices_num, unique_cats_num, lambda_epochs=1, drop=0.2):

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb_size = choices_num
            self.lambda_epochs = lambda_epochs
            self.embeddings = nn.Embedding(unique_cats_num, self.emb_size, max_norm=1, norm_type=2.0)
            self.dropout = nn.Dropout(drop)

            self.fc1 = nn.Linear(emb_vars_num*choices_num, emb_vars_num*choices_num)
            # self.bn1 = nn.BatchNorm1d(emb_vars_num * choices_num)
            self.fc2 = nn.Linear(cont_vars_num*choices_num, cont_vars_num*choices_num)

            self.utilities1 = nn.Conv2d(1, 1, kernel_size=(emb_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)
            self.utilities2 = nn.Conv2d(1, 1, kernel_size=(cont_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)

            self.activation1 = nn.Softplus()
            self.activation2 = nn.Softplus()

        def forward(self, main_input, emb_input):
            emb = self.embeddings(emb_input)
            emb = self.dropout(emb)
            emb = emb.float()
            emb = emb.reshape(-1, emb_vars_num * choices_num)
            emb = self.fc1(emb)
            # emb = self.bn1(emb)
            emb = emb.reshape(-1, 1, emb_vars_num, choices_num)

            main = main_input.float()
            main = main.reshape(-1, cont_vars_num*choices_num)
            main = self.fc2(main)
            main = main.reshape(-1, 1, cont_vars_num, choices_num)

            utilities1 = self.utilities1(emb)
            utilities2 = self.utilities2(main)
            self.utilities1.weight.data.clamp_(min=0)
            output1 = utilities1.reshape(-1, choices_num)
            output2 = utilities2.reshape(-1, choices_num)
            evidence = dict()
            evidence[0] = self.activation1(output1)
            evidence[1] = self.activation2(output2)

            return evidence

    model = Model()

    return model


def TEL_MNL(cont_vars_num, emb_vars_num, choices_num, unique_cats_num,
            extra_emb_dims, n_nodes, lambda_epochs=1, drop=0.2):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb_size = choices_num + extra_emb_dims
            self.lambda_epochs = lambda_epochs
            self.embeddings = nn.Embedding(unique_cats_num, self.emb_size, max_norm=1, norm_type=2.0)
            self.dropout = nn.Dropout(drop)

            self.dense = nn.Conv2d(1, n_nodes, kernel_size=(emb_vars_num*extra_emb_dims, 1), stride=1, padding=0, dtype=torch.float32)
            self.relu3 = nn.ReLU()
            self.fc3 = nn.Linear(n_nodes, choices_num, dtype=torch.float32)

            self.fc1 = nn.Linear(emb_vars_num * choices_num, emb_vars_num * choices_num)
            self.fc2 = nn.Linear(cont_vars_num * choices_num, cont_vars_num * choices_num)
            # self.bn1 = nn.BatchNorm1d(emb_vars_num * choices_num)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

            self.utilities1 = nn.Conv2d(1, 1, kernel_size=(emb_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)
            self.utilities2 = nn.Conv2d(1, 1, kernel_size=(cont_vars_num, 1),
                                        stride=1, padding=0, bias=False, dtype=torch.float32)

            self.activation1 = nn.Softplus()
            self.activation2 = nn.Softplus()
            self.activation3 = nn.Softplus()

        def forward(self, main_input, emb_input):
            emb = self.embeddings(emb_input)
            emb = self.dropout(emb)
            emb = emb.float()

            emb_extra = emb[:, :, choices_num:]
            emb_extra = emb_extra.reshape(-1, 1, emb_vars_num*extra_emb_dims, 1)
            emb_extra = self.dense(emb_extra)
            emb_extra = self.relu3(emb_extra)
            emb_extra = emb_extra.reshape(-1, n_nodes)
            emb_extra = self.fc3(emb_extra)

            emb = emb[:, :, :choices_num]
            emb = emb.reshape(-1, emb_vars_num * choices_num)
            emb = self.fc1(emb)
            # emb = self.bn1(emb)
            emb = self.relu1(emb)
            emb = emb.reshape(-1, 1, emb_vars_num, choices_num)

            main = main_input.float()
            main = main.reshape(-1, cont_vars_num * choices_num)
            main = self.fc2(main)
            main = self.relu2(main)
            main = main.reshape(-1, 1, cont_vars_num, choices_num)

            utilities1 = self.utilities1(emb)
            utilities2 = self.utilities2(main)
            self.utilities1.weight.data.clamp_(min=0.0)
            output1 = utilities1.reshape(-1, choices_num)
            output2 = utilities2.reshape(-1, choices_num)
            evidence = dict()
            evidence[0] = self.activation1(output1)
            evidence[1] = self.activation2(output2)
            evidence[2] = self.activation3(emb_extra)

            return evidence

    model = Model()

    return model

