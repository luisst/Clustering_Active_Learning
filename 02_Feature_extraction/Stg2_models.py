import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_256 as resnet
from torch.nn import Parameter


class background_resnet_ext(nn.Module):
    def __init__(self, num_classes, inter_size=256, backbone='resnet34', pretrained_path=None):
        super(background_resnet_ext, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False)
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.fc_inter1 = nn.Linear(in_features=256, out_features=128)
        nn.init.xavier_uniform_(self.fc_inter1.weight)

        self.fc_final = nn.Linear(in_features=128, out_features=num_classes)
        nn.init.xavier_uniform_(self.fc_final.weight)

        # Dropout layer for fc_final - only used during loss calculation
        self.fc_final_dropout = nn.Dropout(p=0.5)

        self.fc1 = Parameter(torch.Tensor(256, 256))
        nn.init.xavier_uniform_(self.fc1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x) #[batch, 256, *, *]

        # Global Average Pooling
        x = x.flatten(start_dim=2)
        x = x.mean(dim=2)
        x = self.relu(self.pretrained.avg_bn(x))

        x = F.linear(x, self.fc1)

        # New intermediate layer for domain adaptation
        x = self.fc_inter1(x)  # Output: (batch_size, 192)
        x = self.relu(x)  # Add activation
        x = F.dropout(x, p=0.3, training=self.training)  # Optional dropout

        # This becomes your new embeddings for few-shot learning
        spk_embeddings = x

        return spk_embeddings

    def get_fc_final_with_dropout(self):
        """
        Returns fc_final weights and bias with dropout applied during training.
        Only used during loss calculation in prototypical loss.
        """
        if self.training:
            # Apply dropout to the weight matrix during training
            dropped_weight = self.fc_final_dropout(self.fc_final.weight)
            return dropped_weight, self.fc_final.bias
        else:
            # During evaluation, return original weights
            return self.fc_final.weight, self.fc_final.bias

class background_resnet(nn.Module):
    def __init__(self, num_classes, backbone='resnet34', pretrained_path=None):
        super(background_resnet, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False)
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.fc1 = Parameter(torch.Tensor(256, 256))
        nn.init.xavier_uniform_(self.fc1)

        self.weight = Parameter(torch.Tensor(num_classes, 256))
        nn.init.xavier_uniform_(self.weight)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x) #[batch, 256, *, *]


        # Global Average Pooling
        x = x.flatten(start_dim=2)
        x = x.mean(dim=2)
        x = self.relu(self.pretrained.avg_bn(x))

        spk_embedding = F.linear(x, self.fc1)

        return spk_embedding

class SimpleClassifier(nn.Module):
    def __init__(self, dim=256, hidden_dim=128, num_classes=2, dropout_rate=0.4):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.fc_classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        nn.init.xavier_uniform_(self.fc_classifier.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

