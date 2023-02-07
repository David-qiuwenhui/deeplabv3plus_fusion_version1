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

# BatchNorm2d 标准化层的超参数
BN_MOMENTUM = 0.01
EPS = 0.001


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
            nn.BatchNorm2d(out_planes, eps=EPS, momentum=BN_MOMENTUM),
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
        in_planes: int,
        expanded_planes: int,
        out_planes: int,
        kernel: int,
        stride: int,
        activation: str,
        use_se: bool,
        width_multi: float,
    ):
        self.in_planes = self.adjust_channels(in_planes, width_multi)
        self.expanded_planes = self.adjust_channels(expanded_planes, width_multi)
        self.out_planes = self.adjust_channels(out_planes, width_multi)
        self.kernel = kernel
        self.stride = stride
        self.use_se = use_se
        self.use_hs = activation == "HS"

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        # 获取8的整数倍的channels（更大化利用硬件资源和加速训练）
        return _make_divisible(channels * width_multi, 8)


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

        # ******************** 跳跃连接的下采样层 ********************
        if downsample is None:
            if in_planes != out_planes or stride != 1:
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
                BasicBlockNew(
                    in_planes=w, expanded_planes=w * expanded_rate, out_planes=w
                ),
                BasicBlockNew(
                    in_planes=w, expanded_planes=w * expanded_rate, out_planes=w
                ),
                BasicBlockNew(
                    in_planes=w, expanded_planes=w * expanded_rate, out_planes=w
                ),
                BasicBlockNew(
                    in_planes=w, expanded_planes=w * expanded_rate, out_planes=w
                ),
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
                                num_features=c * (2**i), eps=EPS, momentum=BN_MOMENTUM
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
                                    num_features=c * (2**j),
                                    eps=EPS,
                                    momentum=BN_MOMENTUM,
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
                                num_features=c * (2**i), eps=EPS, momentum=BN_MOMENTUM
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


# TODO
class DeepLabV3PlusFusion(nn.Module):
    def __init__(
        self,
        base_channel,
        inverted_residual_setting: List,
    ):
        super().__init__()
        # ******************** Conv1 ********************
        self.conv1 = ConvBNActivation(3, 16, kernel_size=3, stride=2, groups=1)
        self.conv2 = ConvBNActivation(16, 32, kernel_size=3, stride=2, groups=1)

        # ******************** Stage1 ********************
        stage1: List[nn.Module] = []
        stage1_setting = inverted_residual_setting["stage1"]
        for cnf in stage1_setting:
            stage1.append(
                InvertedBotteneck(
                    cnf.in_planes,
                    cnf.expanded_planes,
                    cnf.out_planes,
                    cnf.kernel,
                    cnf.stride,
                    cnf.use_hs,
                    cnf.use_se,
                )
            )
        self.stage1 = nn.Sequential(*stage1)

        # ******************** Transition1 ********************
        self.transition1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=stage1_setting[-1].out_planes,
                        out_channels=base_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(base_channel, eps=EPS, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=stage1_setting[-1].out_planes,
                        out_channels=base_channel * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(base_channel * 2, eps=EPS, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # ******************** Stage2 ********************
        self.stage2 = nn.Sequential(
            StageModule(
                input_branches=2, output_branches=2, c=base_channel, expanded_rate=4
            ),
        )

        # ******************** Transition2 ********************
        self.transition2 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
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
                        num_features=base_channel * 4, eps=EPS, momentum=BN_MOMENTUM
                    ),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # ******************** Stage3 ********************
        self.stage3 = nn.Sequential(
            StageModule(
                input_branches=3, output_branches=3, c=base_channel, expanded_rate=4
            ),
            StageModule(
                input_branches=3, output_branches=3, c=base_channel, expanded_rate=4
            ),
            StageModule(
                input_branches=3, output_branches=3, c=base_channel, expanded_rate=4
            ),
            StageModule(
                input_branches=3, output_branches=3, c=base_channel, expanded_rate=4
            ),
        )

        # ******************** transition3 ********************
        self.transition3 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
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
                        num_features=base_channel * 8, eps=EPS, momentum=BN_MOMENTUM
                    ),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # ******************** Stage4 ********************
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(
                input_branches=4, output_branches=4, c=base_channel, expanded_rate=4
            ),
            StageModule(
                input_branches=4, output_branches=4, c=base_channel, expanded_rate=4
            ),
            StageModule(
                input_branches=4, output_branches=1, c=base_channel, expanded_rate=4
            ),
        )

    def forward(self, x):
        # ******************** Conv1 ********************
        x = self.conv1(x)
        # TODO: 卷积层分开 取第一层级的特征图
        conv1_features = x  # Conv1层的特征图
        x = self.conv2(x)

        # ******************** Stage1 ********************
        x = self.stage1(x)
        stage1_features = x  # Stage1层的特征图

        # ******************** Transition1 ********************
        x = [
            trans(x) for trans in self.transition1
        ]  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8)]

        # ******************** Stage2 ********************
        x = self.stage2(x)  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8)]
        stage2_features = x[0]  # Stage2层的特征图

        # ******************** Transition2 ********************
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1]),
        ]
        # 新的分支由此stage尺度最小的特征下采样和升高维度得到
        # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16)]

        # ******************** Stage3 ********************
        x = self.stage3(x)  # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16)]
        stage3_features = x[0]  # Stage3层的特征图

        # ******************** Transition3 ********************
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]
        # 新的分支由此stage尺度最小的特征下采样和升高维度得到
        # x[x0(B,32,H/4,W/4), x1(B,64,H/8,W/8), x2(B,128,H/16,W/16), x3(B,256,H/32,W/32)]

        # ******************** Stage4 ********************
        x = self.stage4(x)  # x[x0(B,32,H/4,W/4)] 所有分支上采样至(H/4,W/4)后逐像素点相加输出
        stage4_features = x[0]  # Stage4层的特征图

        return (
            conv1_features,
            stage1_features,
            stage2_features,
            stage3_features,
            stage4_features,
            x[0],
        )


def deeplabv3plus_fusion_backbone(model_type):
    if model_type == "hrnet_w18":
        base_channel = 18
    elif model_type == "hrnet_w32":
        base_channel = 32
    elif model_type == "hrnet_w48":
        base_channel = 48

    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)

    # 定义Stage1模块倒残差模块的参数 InvertedResidualConfig
    # in_planes, expanded_planes, out_planes, kernel, stride, activation, use_se, width_multi
    stage1_setting = [
        bneck_conf(32, 128, 32, 3, 1, "RE", False),
        bneck_conf(32, 128, 32, 3, 1, "RE", False),
        bneck_conf(32, 128, 32, 3, 1, "RE", False),
        bneck_conf(32, 128, 32, 3, 1, "RE", False),
    ]
    inverted_residual_setting = dict(stage1=stage1_setting)

    return DeepLabV3PlusFusion(base_channel, inverted_residual_setting)
