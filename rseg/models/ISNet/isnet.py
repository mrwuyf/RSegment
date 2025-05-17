import torch
import torch.nn as nn
import torch.nn.functional as F
from rseg.models.ISNet.imagelevel import ImageLevelContext
from rseg.models.ISNet.semanticlevel import SemanticLevelContext
from rseg.models.backbones import (resnet)
from torchvision.models._utils import IntermediateLayerGetter

class ISNet(nn.Module):
    def __init__(self, num_classes):
        super(ISNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = resnet.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.backbone = IntermediateLayerGetter(self.backbone,
                                                return_layers={"layer4": "res4"})
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.ilc_net = ImageLevelContext(feats_channels=512, transform_channels=256)
        self.slc_net = SemanticLevelContext(feats_channels=512, transform_channels=256)
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1)
        )

        self.decoder_stage2 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = self.backbone(x)
        x = features['res4']
        feats = self.bottleneck(x)
        feats_il = self.ilc_net(feats)
        preds = self.decoder_stage1(feats)
        feats_sl = self.slc_net(feats, preds, feats_il)
        preds_stage2 = self.decoder_stage2(feats_sl)
        preds_stage1 = F.interpolate(preds, (h, w), mode='bilinear', align_corners=False)
        preds_stage2 = F.interpolate(preds_stage2, (h, w), mode='bilinear', align_corners=False)
        if self.training:
            return preds_stage2, preds_stage1
        else:
            return preds_stage2


if __name__ == '__main__':
    from thop import profile
    x = torch.randn(1, 3, 512, 512).cuda()
    net = ISNet(num_classes=6).cuda()
    net.train()
    out, aux = net(x)

    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    print(out.shape)








