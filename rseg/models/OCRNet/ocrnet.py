import torch
import torch.nn as nn
import torch.nn.functional as F
from rseg.models.backbones import hrnet
from geoseg.models.OCRNet.objectcontext import ObjectContextBlock
from geoseg.models.OCRNet.spatialgather import SpatialGatherModule


class OCRNet(nn.Module):
    def __init__(self, num_classes):
        super(OCRNet, self).__init__()
        self.backbone = hrnet.get_hrnetv2_w32()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(480, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.auxiliary_decoder = nn.Sequential(
            nn.Conv2d(480, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.spatial_gather_module = SpatialGatherModule()
        self.object_context_block = ObjectContextBlock(in_channels=512, transform_channels=256)
        self.decoder = nn.Sequential(
            nn.Dropout(0),
            nn.Conv2d(512, num_classes, 1, 1)

        )

    def forward(self, x):
        _, _, h, w = x.size()
        backbone_output = self.backbone(x)
        seg_logits_aux = self.auxiliary_decoder(backbone_output[-1])
        feats = self.bottleneck(backbone_output[-1])
        context = self.spatial_gather_module(feats, seg_logits_aux)
        feats = self.object_context_block(feats, context)
        seg_logits = self.decoder(feats)
        aux = F.interpolate(seg_logits_aux, (h, w), mode='bilinear', align_corners=False)
        pred = F.interpolate(seg_logits, (h, w), mode='bilinear', align_corners=False)
        if self.training:
            return pred, aux
        else:
            return pred

if __name__ == '__main__':
    from thop import profile
    x = torch.randn(1, 3, 512, 512).cuda()
    net = OCRNet(num_classes=6).cuda()
    net.train()
    out, aux = net(x)

    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    print(out.shape)
