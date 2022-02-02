import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.nn.functional as F


model_urls = {
    'resnet18':  'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':  'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':  'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def init_weights(m):
    init.normal(m.weight, std=0.001)
    if m.bias is not None:
        init.constant(m.bias, 0)


def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class Block(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Load weights of pre-trained network
        self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, 4)

        # Split into grading and  counting branches
        self.fc.apply(init_weights)
        self.counting = nn.Linear(512 * block.expansion, 65)
        self.counting.apply(init_weights)

        self.bn1.apply(set_bn_fix)
        self.layer1.apply(set_bn_fix)
        self.layer2.apply(set_bn_fix)
        self.layer3.apply(set_bn_fix)
        self.layer4.apply(set_bn_fix)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x_new):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        cls = self.fc(x)
        cou = self.counting(x)
        cls = F.softmax(cls) + 1e-4
        cou = F.softmax(cou) + 1e-4
        cou2cls = torch.stack((torch.sum(cou[:, :5], 1), torch.sum(cou[:, 5:20], 1), torch.sum(cou[:, 20:50], 1), torch.sum(cou[:, 50:], 1)), 1)

        return cls, cou, cou2cls

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            self.bn1.apply(set_bn_eval)
            self.layer1.apply(set_bn_eval)
            self.layer2.apply(set_bn_eval)
            self.layer3.apply(set_bn_eval)
            self.layer4.apply(set_bn_eval)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Block, [3, 4, 6, 3], **kwargs)
    return model