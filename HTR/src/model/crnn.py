import torch.nn as nn
import torch.nn.functional as F

# From: https://github.com/georgeretsi/HTR-best-practices/


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act_fct=nn.ReLU()):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.act_fct = act_fct

    def forward(self, x):
        out = self.act_fct(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act_fct(out)
        return out


class CNN(nn.Module):
    def __init__(self, cnn_cfg):
        super(CNN, self).__init__()

        self.k = 1

        act_fct = nn.ReLU()

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [2, 2], 3), act_fct])

        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            elif m == 'MH':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x, act_fct=act_fct))

                    in_channels = x
                    cnt += 1

    def forward(self, x):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k // 2])

        return y


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


class CTCHead(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, dropout_last_fc_crnn=0.2, dropout_lstm=0.2):
        super(CTCHead, self).__init__()

        hidden, num_layers = rnn_cfg

        self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=dropout_lstm)
        self.fnl = nn.Sequential(nn.Dropout(dropout_last_fc_crnn), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
        # x dimension: batch size, 256, 1,  nb frames
        y = x.permute(2, 3, 0, 1)[0]  # Output dimension: nb frames, batch size, 256
        y = self.rec(y)[0]

        after_blstm = y

        y = self.fnl(y)  # Output dimension: nb frames, batch size, alphabet size

        y_aux = self.cnn(x).permute(2, 3, 0, 1)[0]

        return [y, y_aux], after_blstm

class CTCtopC(nn.Module):
    def __init__(self, input_size, nclasses, dropout=0.0):
        super(CTCtopC, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.cnn_top = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
    
        x = self.dropout(x)

        y = self.cnn_top(x)
        y = y.permute(2, 3, 0, 1)[0]
        return [y, None], None

class CRNN(nn.Module):
    def __init__(self, cnn_cfg, head_cfg, nclasses, head_type="crnn", dropout_last_fc_crnn=0.2, dropout_lstm=0.2):
        super(CRNN, self).__init__()

        self.features = CNN(cnn_cfg)

        hidden = cnn_cfg[-1][-1]
        if head_type == 'cnn':
            self.top = CTCtopC(hidden, nclasses, dropout=dropout_last_fc_crnn)
        else:
            self.top = CTCHead(hidden, head_cfg, nclasses, dropout_last_fc_crnn, dropout_lstm)

        self.nclasses = nclasses

    def forward(self, x):

        y = self.features(x)  # Output dimension: batch size, 256, 1,  nb frames

        before_blstm = y

        y, after_blstm = self.top(y)

        return y, before_blstm, after_blstm


