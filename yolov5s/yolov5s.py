from model_component import *

import torch
import torch.nn as nn

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class YOLOv5s(nn.Module):
    def __init__(self, num_classes=80, num_anchors=3):
        super(YOLOv5s, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detect_head_channel = self.num_anchors * (self.num_classes + 5)  # 3 * (80 + 5) = 255

        ### backbone
        self.focus_0 = Focus(3, 32, k=3, s=1)
        self.conv_1  = Conv(32, 64, k=3, s=2)
        self.c3_2    = C3(64, 64)
        self.conv_3  = Conv(64, 128, k=3, s=2)
        # self.c3_4_0  = C3(128, 128)
        # self.c3_4_1  = C3(128, 128)
        # self.c3_4_2  = C3(128, 128)
        self.c3_4 = C3(128, 128, n = 3)
        self.conv_5  = Conv(128, 256, k=3, s=2)
        # self.c3_6_0  = C3(256, 256)
        # self.c3_6_1  = C3(256, 256)
        # self.c3_6_2  = C3(256, 256)
        self.c3_6 = C3(256, 256, n = 3)
        self.conv_7  = Conv(256, 512, k=3, s=2)
        self.spp_8   = SPP(512, 512, k=(5, 9, 13))
        self.c3_9    = C3(512, 512, shortcut = False)


        ### detect head
        self.conv_10 = Conv(512, 256, k=1, s = 1)
        self.upsample_11 = nn.Upsample(scale_factor=2, mode='nearest')
        self_concat_12 = None
        self.c3_13 = C3(512, 256, shortcut = False)

        self.conv_14 = Conv(256, 128, k=1, s=1)
        self.upsample_15 = nn.Upsample(scale_factor=2, mode='nearest')
        self_concat_16 = None
        self.c3_17 = C3(256, 128, shortcut = False)

        self.conv_18 = Conv(128, 128, k=3, s=2)
        self_concat_19 = None
        self.c3_20 = C3(256, 256, shortcut = False)

        self.conv_21 = Conv(256, 256, k=3, s=2)
        self_concat_22 = None
        self.c3_23 = C3(512, 512, shortcut = False)

        self.detect_layer_0_conv = nn.Conv2d(128, self.detect_head_channel, 1)
        self.detect_layer_1_conv = nn.Conv2d(256, self.detect_head_channel, 1)
        self.detect_layer_2_conv = nn.Conv2d(512, self.detect_head_channel, 1)

        # self.detect_layer_0_conv = Conv(128, self.detect_head_channel, k=1, s=1)  
        # self.detect_layer_1_conv = Conv(256, self.detect_head_channel, k=1, s=1)
        # self.detect_layer_2_conv = Conv(512, self.detect_head_channel, k=1, s=1)     
        

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.focus_0(x)
        out = self.conv_1(out)
        out = self.c3_2(out)
        out = self.conv_3(out)
        out = self.c3_4(out)
        # out = self.c3_4_1(out)
        # out = self.c3_4_2(out)
        out_4 = out
        out = self.conv_5(out)
        out = self.c3_6(out)
        # out = self.c3_6_1(out)
        # out = self.c3_6_2(out)
        out_6 = out
        out = self.conv_7(out)
        out = self.spp_8(out)
        out = self.c3_9(out)
        out = self.conv_10(out)
        out_10 = out
        out = self.upsample_11(out)
        out = torch.cat((out, out_6), dim=1)
        out = self.c3_13(out)
        out = self.conv_14(out)
        out_14 = out
        out = self.upsample_15(out)
        out = torch.cat((out, out_4), dim=1)
        out = self.c3_17(out)
        out_A = out
        out = self.conv_18(out)
        out = torch.cat((out, out_14), dim=1)
        out = self.c3_20(out)
        out_B = out
        out = self.conv_21(out)
        out = torch.cat((out, out_10), dim=1)       
        out = self.c3_23(out)
        out_C = out

        out_layer_0 = self.detect_layer_0_conv(out_A)
        out_layer_1 = self.detect_layer_1_conv(out_B)
        out_layer_2 = self.detect_layer_2_conv(out_C) 


        return (out_layer_0, out_layer_1, out_layer_2)
    
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                #print('Fusing Conv and BN...')
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        
        return self
