"""
@author: qiuwenhui
@Software: VSCode
@Time: 2023-02-06 23:02:12
"""
from functools import partial
from typing import Callable, List, Optional
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

BN_MOMENTUM = 0.1


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:  # min_ch限制channels的下限值
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)  # '//'取整除
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


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
            nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
            activation_layer(inplace=True),
        )  # inplace=True 不创建新的对象，直接对原始对象进行修改


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(ch=input_c // squeeze_factor, divisor=8)
        self.fc1 = nn.Conv2d(
            input_c, squeeze_c, kernel_size=1
        )  # fc1: expand_channel // 4 (Conv2d 1x1代替全连接层)
        self.fc2 = nn.Conv2d(
            squeeze_c, input_c, kernel_size=1
        )  # fc2: expand_channel (Conv2d 1x1代替全连接层)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # 自适应全局平均池化处理
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x  # SE通道注意力机制与Conv3x3主分支结果相乘


class InvertedResidualConfig:
    def __init__(
        self,
        input_c: int,
        kernel: int,
        expanded_c: int,
        out_c: int,
        use_se: bool,
        activation: str,
        stride: int,
        width_multi: float,
    ):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod
    def adjust_channels(
        channels: int, width_multi: float
    ):  # 获取8的整数倍的channels（更大化利用硬件资源和加速训练）
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
    ):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 检测是否使用shortcu捷径分支（stride=1不进行下采样 and input_c==output_c）
        self.use_res_connect = cnf.stride == 1 and cnf.input_c == cnf.out_c

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # 使用conv2d 1*1卷积模块进行升维操作
        # Expand block
        if cnf.expanded_c != cnf.input_c:
            layers.append(
                ConvBNActivation(
                    cnf.input_c,
                    cnf.expanded_c,
                    kernel_size=1,
                    activation_layer=activation_layer,
                )
            )

        # Depthwise block 逐通道卷积Depthwise Conv
        layers.append(
            ConvBNActivation(
                cnf.expanded_c,
                cnf.expanded_c,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=cnf.expanded_c,
                activation_layer=activation_layer,
            )
        )

        # SqueezeExcitation attention block
        if cnf.use_se:  # 使用SE通道注意力机制
            layers.append(SqueezeExcitation(cnf.expanded_c))
            # SqueezeExcitation(AdaptiveAvgPool->fc1->ReLU->fc2->hardsigmoid  input*SE_result

        # Project block 逐点卷积Pointwise Conv
        layers.append(
            ConvBNActivation(
                cnf.expanded_c,
                cnf.out_c,
                kernel_size=1,
                activation_layer=nn.Identity,
            )
        )  # nn.Identity 线性激活

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        expanded_planes,
        out_planes,
        kernel=3,
        stride=1,
        use_hs=False,
        use_se=False,
        downsample=None,
    ):
        super().__init__()

        # ******************** 非线性激活层 ********************
        self.activation_layer = nn.Hardswish if use_hs else nn.ReLU6
        self.activation = self.activation_layer(inplace=True)

        # ******************** shortcut连接的下采样层 ********************
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

        # ******************** 主分支通路 ********************
        layers: List[nn.Module] = []
        layers.append(
            ConvBNActivation(
                in_planes,
                expanded_planes,
                kernel_size=1,
                stride=1,
                activation_layer=self.activation_layer,
            )
        )
        layers.append(
            ConvBNActivation(
                expanded_planes,
                expanded_planes,
                kernel_size=kernel,
                stride=stride,
                groups=expanded_planes,
                activation_layer=self.activation_layer,
            )
        )
        # 引入通道注意力机制
        if use_se:
            layers.append(SqueezeExcitation(input_c=expanded_planes, squeeze_factor=4))
        layers.append(
            ConvBNActivation(
                expanded_planes,
                out_planes,
                kernel_size=1,
                stride=1,
                activation_layer=nn.Identity,
            )
        )

        self.block = nn.Sequential(*layers)
        self.outchannels = out_planes
        self.is_stride = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.block(x)
        residual = self.downsample(residual)
        out += residual
        out = self.activation(out)

        return out


class InvertedBotteneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        expanded_planes,
        out_planes,
        kernel=3,
        stride=1,
        use_hs=False,
        use_se=False,
        downsample=None,
    ):
        super().__init__()

        # ******************** 非线性激活层 ********************
        self.activation_layer = nn.Hardswish if use_hs else nn.ReLU6
        self.activation = self.activation_layer(inplace=True)

        # ******************** shortcut连接的下采样层 ********************
        if downsample is None:
            if in_planes != expanded_planes and stride != 1:
                self.downsample = ConvBNActivation(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    activation_layer=nn.Identity,
                )
            else:
                self.downsample = nn.Identity()
        else:
            self.downsample = downsample

        # ******************** 主分支通路 ********************
        layers: List[nn.Module] = []
        layers.append(
            ConvBNActivation(
                in_planes,
                expanded_planes,
                kernel_size=1,
                stride=1,
                activation_layer=self.activation_layer,
            )
        )
        layers.append(
            ConvBNActivation(
                expanded_planes,
                expanded_planes,
                kernel_size=kernel,
                stride=stride,
                groups=expanded_planes,
                activation_layer=self.activation_layer,
            )
        )
        # 引入通道注意力机制
        if use_se:
            layers.append(SqueezeExcitation(input_c=expanded_planes, squeeze_factor=4))
        layers.append(
            ConvBNActivation(
                expanded_planes,
                out_planes,
                kernel_size=1,
                stride=1,
                activation_layer=nn.Identity,
            )
        )

        self.block = nn.Sequential(*layers)
        self.outchannels = out_planes
        self.is_stride = stride > 1
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.block(x)
        residual = self.downsample(residual)
        out += residual
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
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


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c, expanded_rate):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2**i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlockNew(in_planes=w, expanded_planes=w * 4, out_planes=w),
                BasicBlockNew(in_planes=w, expanded_planes=w * 4, out_planes=w),
                BasicBlockNew(in_planes=w, expanded_planes=w * 4, out_planes=w),
                BasicBlockNew(in_planes=w, expanded_planes=w * 4, out_planes=w),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=c * (2**j),
                                out_channels=c * (2**i),
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_features=c * (2**i), momentum=BN_MOMENTUM
                            ),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode="bilinear"),
                        )
                    )
                else:
                    # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    in_channels=c * (2**j),
                                    out_channels=c * (2**j),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(
                                    num_features=c * (2**j), momentum=BN_MOMENTUM
                                ),
                                nn.ReLU(inplace=True),
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=c * (2**j),
                                out_channels=c * (2**i),
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_features=c * (2**i), momentum=BN_MOMENTUM
                            ),
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum(
                        [
                            self.fuse_layers[i][j](x[j])
                            for j in range(len(self.branches))
                        ]
                    )
                )
            )

        return x_fused


class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 17):
        super().__init__()
        # Stem 使用两个卷积模块进行四倍下采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(inplanes=64, planes=64, downsample=downsample),
            Bottleneck(inplanes=256, planes=64),
            Bottleneck(inplanes=256, planes=64),
            Bottleneck(inplanes=256, planes=64),
        )

        self.transition1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=base_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=base_channel * 2,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                ),
            ]
        )

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2
        self.transition2 = nn.ModuleList(
            [
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=base_channel * 2,
                            out_channels=base_channel * 4,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(
                            num_features=base_channel * 4, momentum=BN_MOMENTUM
                        ),
                        nn.ReLU(inplace=True),
                    )
                ),
            ]
        )

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
        )

        # transition3
        self.transition3 = nn.ModuleList(
            [
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Identity(),  # None,  - Used in place of "None" because it is callable
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=base_channel * 4,
                            out_channels=base_channel * 8,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(
                            num_features=base_channel * 8, momentum=BN_MOMENTUM
                        ),
                        nn.ReLU(inplace=True),
                    )
                ),
            ]
        )

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=1, c=base_channel),
        )

    def forward(self, x):
        x = self.conv1(x)  # x(B,3,H,W) -> x(B,64,H/2,W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # x(B,64,H/4,W/4)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)  # x(B,256,H/4,W/4)
        low_level_features = x  # 浅层的特征图

        x = [
            trans(x) for trans in self.transition1
        ]  # Since now, x is a list  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8)]

        x = self.stage2(x)  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8)]
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1]),
        ]  # New branch derives from the "upper" branch only
        # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16)]

        x = self.stage3(x)  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16)]
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only
        # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16), x3(B,256,H/32,W/32)]

        x = self.stage4(x)  # x[x0(B,32,H/4,W/4)] 所有分支上采样至(H/4,W/4)后逐像素点相加输出

        return low_level_features, x[0]


def HRNet_Backbone_New(model_type):
    if model_type == "hrnet_w18":
        backbone = HighResolutionNet(base_channel=18)
    elif model_type == "hrnet_w32":
        backbone = HighResolutionNet(base_channel=32)
    elif model_type == "hrnet_w48":
        backbone = HighResolutionNet(base_channel=48)

    return backbone


class DeepLabV3PlusFusion(nn.Module):
    def __init__(
        self,
        base_channel,
        inverted_residual_setting: List,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # ******************** Conv1 ********************
        self.conv1 = nn.Sequential(
            ConvBNActivation(3, 16, kernel_size=3, stride=2, groups=1),
            ConvBNActivation(16, 16, kernel_size=3, stride=1, groups=1),
        )

        # ******************** Stage1 ********************
        stage1: List[nn.Module] = []
        stage1_setting = inverted_residual_setting["stage1"]
        for cnf in stage1_setting:
            stage1.append(block(cnf, norm_layer))
        self.stage1 = nn.Sequential(*stage1)

        # TODO
        # TODO
        # TODO

        self.stage1 = nn.Sequential(
            InvertedBotteneck(
                16,
            )
        )

        # ******************** Transition1 ********************
        self.transition1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=stage1_setting[-1].out_c,
                        out_channels=base_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=base_channel * 2,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                ),
            ]
        )

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )


def deeplabv3plus_fusion_backbone():
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)

    # 定义Stage1模块倒残差模块的参数 InvertedResidualConfig
    # input_c, kernel, expanded_c, out_c, use_se, activation, stride
    stage1_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 3, 72, 40, False, "RE", 1),
    ]

    inverted_residual_setting = dict(stage1=stage1_setting)

    return DeepLabV3PlusFusion(32, inverted_residual_setting)
