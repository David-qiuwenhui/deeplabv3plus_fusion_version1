from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.deeplabv3plus_fusion import deeplabv3plus_fusion_backbone
from nets.hrnet import HRNet_Backbone, hrnet_classification
from nets.hrnet_new import HRNet_Backbone_New
from nets.mobilenetv3 import mobilenet_v3_large_backbone
from nets.mobilevit import mobile_vit_small_backbone
from nets.repvgg_new import repvgg_backbone_new, repvgg_model_convert
from nets.resnet import resnet50_backbone
from nets.resnext import resnext50_32x4d_backbone
from nets.swin_transformer import Swin_Transformer_Backbone
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

# BatchNorm2d 标准化层的超参数
BN_MOMENTUM = 0.01
EPS = 0.001


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        padding = (kernel_size - 1) // 2  # 取整除
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes, eps=EPS, momentum=BN_MOMENTUM),
            activation_layer(inplace=True),
        )  # inplace=True 不创建新的对象，直接对原始对象进行修改


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True, downsample_factor=8):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(
        self, dim_in, dim_out, rate=1, bn_mom=0.1
    ):  # dim_in=2048, dim_out=256, rate=2
        super(ASPP, self).__init__()
        # Conv1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=6 * 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=6 * rate,
                dilation=6 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=12 * 2
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=12 * rate,
                dilation=12 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=18 * 2
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=18 * rate,
                dilation=18 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv1x1 branch 全局平均池化层
        self.branch5_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.branch5_bn = nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        # 对ASPP模块concat后的结果进行卷积操作（降低维度）
        self.conv_cat = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_out * 5,
                out_channels=dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()  # x(bs,2048,64,64)
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)  # conv1x1(bs, 256, 64, 64)
        conv3x3_1 = self.branch2(x)  # conv3x3_1(bs, 256, 64, 64)
        conv3x3_2 = self.branch3(x)  # conv3x3_2(bs, 256, 64, 64)
        conv3x3_3 = self.branch4(x)  # conv3x3_3(bs, 256, 64, 64)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(
            input=x, dim=2, keepdim=True
        )  # global_feature(bs, 2048, 1, 64)
        global_feature = torch.mean(
            input=global_feature, dim=3, keepdim=True
        )  # global_feature(bs, 2048, 1, 1)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(
            global_feature
        )  # global_feature(bs, 256, 1, 1)
        global_feature = F.interpolate(
            input=global_feature,
            size=(row, col),
            scale_factor=None,
            mode="bilinear",
            align_corners=True,
        )  # global_feature(bs, 256, 64, 64)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征
        # -----------------------------------------#
        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1
        )  # feature_cat(bs, 1280, 64, 64)
        result = self.conv_cat(feature_cat)  # result(bs, 256, 64, 64)
        return result


class DeepLab(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone,
        pretrained=False,
        downsample_factor=8,
        backbone_path="",
    ):
        super(DeepLab, self).__init__()
        self.backbone_name = backbone
        if backbone == "deeplabv3_fusion":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [32,H/4,W/4]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = deeplabv3plus_fusion_backbone(model_type="hrnet_w32")
            in_channels = 32
            # low_level_channels = 32
            # 浅层特征的通道数量
            conv1_channels = 16
            stage1_channels = 32
            stage2_channels = 32
            stage3_channels = 32
            stage4_channels = 32

            # ASPP模块融合后输出的通道数量
            aspp_channels = 256
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use mobilenet, xception.".format(backbone)
            )

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(
            dim_in=in_channels, dim_out=aspp_channels, rate=16 // downsample_factor
        )  # dim_in=2048 dim_out=256 rate=2

        # ----------------------------------#
        #   浅、中层特征图的卷积传递层
        # ----------------------------------#
        self.shortcut_conv0 = ConvBNActivation(
            in_planes=conv1_channels,
            out_planes=conv1_channels,
            kernel_size=1,
            stride=1,
        )

        self.shortcut_conv1 = ConvBNActivation(
            in_planes=stage1_channels,
            out_planes=stage1_channels,
            kernel_size=1,
            stride=1,
        )

        self.shortcut_conv2 = ConvBNActivation(
            in_planes=stage2_channels,
            out_planes=stage2_channels,
            kernel_size=1,
            stride=1,
        )

        self.shortcut_conv3 = ConvBNActivation(
            in_planes=stage3_channels,
            out_planes=stage3_channels,
            kernel_size=1,
            stride=1,
        )

        self.shortcut_conv4 = ConvBNActivation(
            in_planes=stage4_channels,
            out_planes=stage4_channels,
            kernel_size=1,
            stride=1,
        )

        # ----------------------------------#
        #   第一阶段的深浅层次特征图卷积处理模块
        # ----------------------------------#
        self.cat_conv1 = nn.Sequential(
            ConvBNActivation(
                in_planes=aspp_channels + stage1_channels * 4,
                out_planes=256,
                kernel_size=3,
            ),
            ConvBNActivation(
                in_planes=256,
                out_planes=256,
                kernel_size=3,
            ),
        )

        # ----------------------------------#
        #   第二阶段的深浅层次特征图卷积处理模块
        # ----------------------------------#
        self.cat_conv2 = ConvBNActivation(
            in_planes=256 + conv1_channels,
            out_planes=256,
            kernel_size=3,
        )

        # ----------------------------------#
        #   DeepLabV3Plus Head 将特征图转换为N个类别预测掩码图像
        # ----------------------------------#
        # 更改channels至num_classes
        self.cls_conv = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1, stride=1
        )

    def forward(self, x):
        H, W = x.size(2), x.size(3)  # x(B,3,H,W)
        # -----------------------------------------#
        #   主干特征提取网络
        # -----------------------------------------#
        if self.backbone_name in ["deeplabv3_fusion"]:
            (
                conv1_features,
                stage1_features,
                stage2_features,
                stage3_features,
                stage4_features,
                x,
            ) = self.backbone(x)

        # -----------------------------------------#
        #   膨胀卷积池化金字塔模块
        # -----------------------------------------#
        x = self.aspp(x)  # x(B,256,H/4,W/4)

        # -----------------------------------------#
        #   浅中层特征图的传递和卷积处理
        # -----------------------------------------#
        conv1_features = self.shortcut_conv0(conv1_features)
        stage1_features = self.shortcut_conv1(stage1_features)
        stage2_features = self.shortcut_conv2(stage2_features)
        stage3_features = self.shortcut_conv3(stage3_features)
        stage4_features = self.shortcut_conv4(stage4_features)

        # -----------------------------------------#
        #   第一阶段的深浅层特征融合
        #   将主分支的加强特征进行上采样
        #   加强特征与浅层特征拼接后再利用卷积进行特征融合
        # -----------------------------------------#
        x = F.interpolate(
            input=x,
            size=(stage1_features.size(2), stage1_features.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = self.cat_conv1(
            torch.cat(
                (x, stage1_features, stage2_features, stage3_features, stage4_features),
                dim=1,
            )
        )

        # -----------------------------------------#
        #   第二阶段的深浅层特征融合
        #   将主分支的特征进行上采样
        #   主分支的特征与浅层特征拼接后再利用卷积进行特征融合
        # -----------------------------------------#
        x = F.interpolate(
            input=x,
            size=(conv1_features.size(2), conv1_features.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = self.cat_conv2(torch.cat((x, conv1_features), dim=1))

        # -----------------------------------------#
        #   特征上采样至原图大小
        # -----------------------------------------#
        x = self.cls_conv(x)  # x(bs, num_classes, H/4, W/4)
        x = F.interpolate(
            input=x, size=(H, W), mode="bilinear", align_corners=True
        )  # x(B, N, H, W)
        return x

    def switch_to_deploy(self):
        if self.backbone_name in ["repvgg_new"]:
            self.backbone = repvgg_model_convert(model=self.backbone)
            print(
                f"\033[1;33;44m 🔬🔬🔬🔬 Switch {self.backbone_name} to deploy model \033[0m"
            )
        else:
            print(f"\033[1;31;41m 🔬🔬🔬🔬 Can not switch to deploy model \033[0m")
