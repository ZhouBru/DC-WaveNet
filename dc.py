import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channel = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output



class CausalConv1d(nn.Module):
    """
    Input and output sizes will be the same.
    """
    def __init__(self, in_size, out_size, kernel_size, dilation=1, groups=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad,
                               dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad]
        return x


class ResidualLayer(nn.Module):
    def __init__(self, residual_size, skip_size, dilation):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_size, residual_size,
                                        kernel_size=3, dilation=dilation)
        self.bn_filter = nn.BatchNorm1d(residual_size)
        self.conv_gate = CausalConv1d(residual_size, residual_size,
                                      kernel_size=3, dilation=dilation)
        self.bn_gate = nn.BatchNorm1d(residual_size)
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        
        self.res_bn = nn.BatchNorm1d(residual_size)
        self.skipconv1_1 = nn.Conv1d(skip_size, skip_size, kernel_size=1)


    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_filter = self.bn_filter(conv_filter)
        
        conv_gate = self.conv_gate(x)
        conv_gate = self.bn_gate(conv_gate)

        activation = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        # activation = self.res_bn(activation) 

        fx = self.resconv1_1(activation)
        #
        skip = self.skipconv1_1(fx)
        #
        
        residual = fx + x
        # residual=[batch,residual_size,seq_len]  skip=[batch,skip_size,seq_len]
        return skip, residual


class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth):
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_size, skip_size, 2 ** layer)
                          for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)

    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
            # skip =[1,batch,skip_size,seq_len]
        return torch.cat(skips, dim=0), x  # [layers,batch,skip_size,seq_len]

    
    
    
class Wavegram(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(Wavegram, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.BatchNorm1d(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(313, 313, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = out.transpose(1,2)
        out = self.conv_encoder(out)
        out = out.transpose(1,2)
        return out
            
        
        
class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)  

class WaveNet(nn.Module):
    def __init__(self, input_size=128, out_size=128, residual_size=512, skip_size=512,
                 dilation_cycles=2, dilation_depth=4):

        super(WaveNet, self).__init__()
        self.input_conv = CausalConv1d(input_size, residual_size, kernel_size=2)
        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(residual_size, skip_size, dilation_depth)
             for cycle in range(dilation_cycles)]
        )
        
        

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.fusion_mode = MS_CAM(channels=dilation_cycles*dilation_depth,r=7)

    def forward(self, x, label=None):
        x = self.input_conv(x)  # [batch,residual_size, seq_len]
        skip_connections = []

        for cycle in self.dilated_stacks:
            skips, x = cycle(x)
            skip_connections.append(skips)
        skip_connections = torch.cat(skip_connections, dim=0)
        
        skip_connections=self.fusion_mode(skip_connections.transpose(1,0))
        skip_connections = skip_connections.transpose(1,0)

        out = skip_connections.sum(dim=0)  
        return out


class DualPath(nn.Module):
    def __init__(self,input_size1=313, out_size1=256, residual_size1=1024, skip_size1=1024,
                 input_size2=128, out_size2=128, residual_size2=512, skip_size2=512, num_classes=23):
# 
        super(DualPath, self).__init__()
        
        self.conv1 = ConvBlock(2, 32, 1, 1, 0)
        self.dw_conv1 = ConvBlock(32, 32, 1, 1, 0, dw=True)
        self.conv2 = ConvBlock(32, 1, 1, 1, 0)

        self.post1 = nn.Sequential(#
            nn.BatchNorm1d(skip_size1),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_size1, skip_size1, kernel_size=3, stride=1, padding=1, groups=skip_size1),
            nn.BatchNorm1d(skip_size1),
            nn.Conv1d(skip_size1, out_size1, 1),
        )
        
        self.post = nn.Sequential(#
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(32,32), stride=(16,16)),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, 1),
        )

        self.tgramnet = Wavegram(mel_bins=128, win_len=1024, hop_len=512)
        self.arcface = ArcMarginProduct(465, num_classes, m=0.7, s=30, sub=2)
        self.firstwave = WaveNet(input_size=input_size1, out_size=out_size1, residual_size=residual_size1, skip_size=skip_size1,
                 dilation_cycles=1, dilation_depth=7)
        self.secondwave = WaveNet(input_size=input_size2, out_size=out_size2, residual_size=residual_size2, skip_size=skip_size2,
                 dilation_cycles=2, dilation_depth=4)
        
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x_wav, x_mel, label=None):
        
        x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)
        x = self.conv1(x)
        x=self.dw_conv1(x)
        x=self.conv2(x)
        x=x.squeeze(1) 

        x = x.transpose(1,2)
        x = self.firstwave(x)
        x = self.post1(x)
        
        x = x.transpose(1,2)
        x = self.secondwave(x)
        x = x.unsqueeze(1)
        x = self.post(x)
        out = x.squeeze(1)
        feature = out.view(out.size(0), -1)
        if label is None:
            return feature
        out = self.arcface(feature, label)
        return out, feature	