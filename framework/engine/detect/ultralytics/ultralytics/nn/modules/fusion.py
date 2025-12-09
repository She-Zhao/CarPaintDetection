"""Fusion modules."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .conv import autopad

__all__ = (
    "ConvLeakyRelu2d",
    "RGBD",
    "ECA", 
    "ConvBnLeakyRelu2d", 
    "MultiScaleAttention",
    "StripeFeatureFusion", 
    "CrossSourceFusion",
    "DefectAwareFusionModule",
    "DefectAwareFusionModule_old",
    "DualStageFeatureFusion",
    "StripeFeatureSelection",
    "CrossSourceFeatureInjection",
    "CrossChannelAttention",
    "CrossSpatialAttention", 
    "DualCBAM",
)


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, padding=p, stride=s, dilation=d, groups=g)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2, inplace=True)
    
# class ConvLeakyReLU(nn.Module):
#     default_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)  # default activation
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         # self.bn = nn.BatchNorm2d(c2)
#         self.bn = nn.Identity()
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))


class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, padding=p, stride=s, dilation=d, groups=g)
        self.bn   = nn.BatchNorm2d(c2)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2, inplace=True)


class ConvBnTanh2d(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, padding=p, stride=s, dilation=d, groups=g)
        self.bn   = nn.BatchNorm2d(c2)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5


class DenseBlock(nn.Module):
    def __init__(self,c1):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(c1, c1)
        self.conv2 = ConvLeakyRelu2d(2*c1, c1)
        # self.conv3 = ConvLeakyRelu2d(3*c1, c1)
    def forward(self,x):
        x = torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


class Sobelxy(nn.Module):
    def __init__(self, c1, k=3, s=1, p=1, g=1, d=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(c1, c1, kernel_size=k, padding=p, stride=s, dilation=d, groups=c1,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        # self.convx.weight.requires_grad = False
        self.convy=nn.Conv2d(c1, c1, kernel_size=k, padding=p, stride=s, dilation=d, groups=c1,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        # self.convy.weight.requires_grad = False
    def forward(self, x):
        return torch.abs(self.convx(x)) + torch.abs(self.convy(x))


class Conv1(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, d=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, padding=p, stride=s, dilation=d, groups=g)
    def forward(self,x):
        return self.conv(x)

    
class RGBD(nn.Module):
    def __init__(self, c1, c2):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(c1)
        self.convdown=Conv1(3*c1,c2)
        self.sobelconv=Sobelxy(c1)
        self.convup =Conv1(c1,c2)
    def forward(self,x):
        return F.leaky_relu(self.convdown(self.dense(x))+self.convup(self.sobelconv(x)),
                            negative_slope=0.1, inplace=True)


class ECA(nn.Module):
    def __init__(self, c1, c2, k=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        src1_ch = [16,32,48]
        src2_ch = [16,32,48]
        output=1
        self.src1_conv=ConvLeakyRelu2d(1,src1_ch[0])
        self.src1_rgbd1=RGBD(src1_ch[0], src1_ch[1])
        self.src1_rgbd2 = RGBD(src1_ch[1], src1_ch[2])
        # self.src1_rgbd3 = RGBD(src1_ch[2], src1_ch[3])
        self.src2_conv=ConvLeakyRelu2d(1, src2_ch[0])
        self.src2_rgbd1 = RGBD(src2_ch[0], src2_ch[1])
        self.src2_rgbd2 = RGBD(src2_ch[1], src2_ch[2])
        # self.src2_rgbd3 = RGBD(src2_ch[2], src2_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(src1_ch[3]+src2_ch[3], src1_ch[2]+src2_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(src1_ch[2]+src2_ch[2], src1_ch[1]+src2_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(src1_ch[1]+src2_ch[1], src1_ch[0]+src2_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(src1_ch[0]+src2_ch[0], src1_ch[0])
        self.decode1 = ConvBnTanh2d(src1_ch[0], output)

    def forward(self, img1, img2):
        # encode
        # x_src1_p=self.src1_conv(img1)
        # x_src1_p1=self.src1_rgbd1(x_src1_p)
        # x_src1_p2=self.src1_rgbd2(x_src1_p1)
        # # x_src1_p3=self.src1_rgbd3(x_src1_p2)
        
        # x1_p=self.src1_rgbd2(self.src1_rgbd1(self.src1_conv(img1)))

        # x_src2_p=self.src2_conv(img2)
        # x_src2_p1=self.src2_rgbd1(x_src2_p)
        # x_src2_p2=self.src2_rgbd2(x_src2_p1)
        # # x_src2_p3=self.src2_rgbd3(x_src2_p2)
        
        # x2_p=self.src2_rgbd2(self.src2_rgbd1(self.src2_conv(img2)))
        
        # decode
        # x=self.decode4(torch.cat((x1_p,x2_p),dim=1))
        x=self.decode4(torch.cat((self.src1_rgbd2(self.src1_rgbd1(self.src1_conv(img1))),
                                  self.src2_rgbd2(self.src2_rgbd1(self.src2_conv(img2)))),dim=1))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x


class FusionDecoder(nn.Module):
    """
    Fusion Decoder for reconstructing fused output from encoded features.
    """
    def __init__(self, output=1):
        super(FusionDecoder, self).__init__()
        src1_ch = [16, 32, 48]
        src2_ch = [16, 32, 48]
        
        # Decoder layers
        self.decode4 = ConvBnLeakyRelu2d(src1_ch[2] + src2_ch[2], src1_ch[1] + src2_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(src1_ch[1] + src2_ch[1], src1_ch[0] + src2_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(src1_ch[0] + src2_ch[0], src1_ch[0])
        self.decode1 = ConvBnTanh2d(src1_ch[0], output)

    def forward(self, fused_features):
        x = self.decode4(fused_features)
        x = self.decode3(x)
        x = self.decode2(x)
        x = self.decode1(x)
        return x


class DynamicFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 边缘注意力生成
        self.edge_attn = nn.Sequential(
            nn.Conv2d(channels, channels//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels//8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 通道重加权
        self.channel_attn = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, stripe_feat, phase_feat):
        # 空间注意力
        spatial_w = self.edge_attn(stripe_feat)  # [B,1,H,W]
        
        # 通道注意力
        channel_w = torch.sigmoid(self.fc(
            self.channel_attn(phase_feat).flatten(1)
        )).view(-1, stripe_feat.size(1), 1, 1)  # [B,C,1,1]
        
        # 动态融合
        return phase_feat + stripe_feat * spatial_w * channel_w


class MultiStripeFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 3D卷积处理时序
        self.conv3d = nn.Conv3d(4, 1, kernel_size=(4,1,1))  # 压缩4幅图→1幅
        
    def forward(self, stripe_feats):
        # stripe_feats: [B,4,C,H,W]
        # 沿时序维度压缩
        compressed = self.conv3d(stripe_feats.permute(0,2,1,3,4))  # [B,C,1,H,W]
        return compressed.squeeze(2)  # [B,C,H,W]


class MultiScaleAttention(nn.Module):
    """多尺度注意力模块，用于处理不同感受野的特征"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        
        self.fc = nn.Sequential(
            nn.Linear(channels * 21, channels // 4),  # 1+4+16=21
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 多尺度池化
        y1 = self.avg_pool1(x).view(b, c)
        y2 = self.avg_pool2(x).view(b, c * 4)
        y4 = self.avg_pool4(x).view(b, c * 16)
        
        # 拼接并生成注意力权重
        y = torch.cat([y1, y2, y4], dim=1)
        attention = self.fc(y).view(b, c, 1, 1)
        
        return x * attention


class StripeFeatureFusion(nn.Module):
    """同源融合模块：融合4个正弦条纹特征图"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 多尺度特征提取
        self.conv3x3 = ConvBnLeakyRelu2d(channels, channels, k=3, p=1)
        self.conv5x5 = ConvBnLeakyRelu2d(channels, channels, k=5, p=2)
        self.conv7x7 = ConvBnLeakyRelu2d(channels, channels, k=7, p=3)
        
        # 下采样-上采样路径，增大感受野（使用自适应池化避免尺寸问题）
        self.conv_large = ConvBnLeakyRelu2d(channels, channels, k=3, p=1)
        
        # 多尺度注意力
        self.ms_attention = MultiScaleAttention(channels)
        
        # 特征重要性学习
        self.feature_weights = nn.Parameter(torch.ones(4) / 4)  # 4个特征图的权重
        self.softmax = nn.Softmax(dim=0)
        
        # 边缘检测，用于抑制背景条纹
        self.edge_detector = Sobelxy(channels)
        self.edge_suppress = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, 1, 1),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.final_conv = ConvBnLeakyRelu2d(channels, channels)
        
    def forward(self, stripe_features):
        # stripe_features: list of 4 tensors, each with shape [B, C, H, W]
        b, c, h, w = stripe_features[0].size()
        
        # 1. 自适应权重融合
        weights = self.softmax(self.feature_weights)
        weighted_sum = sum(feat * weight for feat, weight in zip(stripe_features, weights))
        
        # 2. 多尺度特征提取
        feat_3x3 = self.conv3x3(weighted_sum)
        feat_5x5 = self.conv5x5(weighted_sum)
        feat_7x7 = self.conv7x7(weighted_sum)
        
        # 3. 大感受野特征（使用自适应池化避免尺寸问题）
        original_size = weighted_sum.shape[-2:]
        # 计算下采样目标尺寸
        target_h = max(1, original_size[0] // 2)
        target_w = max(1, original_size[1] // 2)
        
        # 自适应下采样
        feat_down = F.adaptive_avg_pool2d(weighted_sum, (target_h, target_w))
        feat_down = self.conv_large(feat_down)
        
        # 自适应上采样回原始尺寸
        feat_up = F.interpolate(feat_down, size=original_size, mode='bilinear', align_corners=True)
        
        # 4. 多尺度特征融合
        multi_scale_feat = feat_3x3 + feat_5x5 + feat_7x7 + feat_up
        
        # 5. 边缘抑制（抑制背景条纹干扰）
        edge_map = self.edge_detector(weighted_sum)
        edge_mask = self.edge_suppress(edge_map)
        # 在边缘区域降低特征强度
        edge_suppressed = multi_scale_feat * (1 - edge_mask * 0.3)
        
        # 6. 多尺度注意力
        attended_feat = self.ms_attention(edge_suppressed)
        
        # 7. 最终融合
        output = self.final_conv(attended_feat)
        
        return output


class CrossSourceFusion(nn.Module):
    """异源融合模块：融合绝对相位特征和正弦融合特征"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 绝对相位特征增强
        self.phase_enhance = nn.Sequential(
            ConvBnLeakyRelu2d(channels, channels),
            ECA(channels, channels)
        )
        
        # 正弦特征适配
        self.stripe_adapt = ConvBnLeakyRelu2d(channels, channels)
        
        # 渐进式融合门控 - 更清晰的实现
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 从绝对相位计算，用于引导正弦特征
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 通道注意力 - 从绝对相位计算，用于引导正弦特征
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.final_fusion = ConvBnLeakyRelu2d(channels, channels)
        
        # 渐进式融合权重 - 控制绝对相位和正弦特征的比例
        self.residual_weight = nn.Parameter(torch.tensor(0.8))  # 保证绝对相位特征的主导地位
        
    def forward(self, phase_feat, stripe_feat):
        # phase_feat: 绝对相位特征 [B, C, H, W]
        # stripe_feat: 正弦融合特征 [B, C, H, W]
        
        # 1. 绝对相位特征增强
        enhanced_phase = self.phase_enhance(phase_feat)
        
        # 2. 正弦特征适配
        adapted_stripe = self.stripe_adapt(stripe_feat)
        
        # 3. 渐进式融合门控
        concat_feat = torch.cat([enhanced_phase, adapted_stripe], dim=1)
        fusion_gate = self.fusion_gate(concat_feat)
        
        # 4. 空间注意力
        spatial_attn = self.spatial_attention(enhanced_phase)
        
        # 5. 通道注意力
        channel_attn = self.channel_attention(enhanced_phase)
        
        # 6. 注意力引导的融合
        attended_stripe = adapted_stripe * spatial_attn * channel_attn
        
        # 7. 渐进式融合（保证绝对相位特征的主导地位）
        fused_feat = enhanced_phase * self.residual_weight + attended_stripe * fusion_gate * (1 - self.residual_weight)
        
        # 8. 最终融合
        output = self.final_fusion(fused_feat)
        
        return output


class StripeFeatureSelection(nn.Module):
    """同源特征选择模块：从4个正弦条纹特征中选择1个最优特征"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 特征质量评估网络 - 评估每个正弦特征的质量
        self.quality_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征选择门控 - 生成one-hot选择向量
        self.selection_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, 16, 1),  # 4个特征拼接
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 1),  # 输出4个选择权重
            nn.Softmax(dim=1)
        )
        
        # 特征增强 - 对选中的特征进行增强
        self.feature_enhance = ConvBnLeakyRelu2d(channels, channels)
        
    # def forward(self, stripe_features):
    #     # stripe_features: list of 4 tensors, each with shape [B, C, H, W]
    #     # 消融版本：不做同源特征选择与门控，直接对4个条纹特征做简单平均
    #     # 尽量少改动，保持返回张量的形状与后续模块兼容
    #     stacked = torch.stack(stripe_features, dim=0)  # [4, B, C, H, W]
    #     selected_feat = stacked.mean(dim=0)  # [B, C, H, W]
    #     return selected_feat
    
    def forward(self, stripe_features):
        # stripe_features: list of 4 tensors, each with shape [B, C, H, W]
        b, c, h, w = stripe_features[0].size()
        
        # 1. 特征质量评估 - 评估每个特征的质量分数
        # quality_scores = []
        # for feat in stripe_features:
        #     quality = self.quality_estimator(feat)  # [B, 1, 1, 1]
        #     quality_scores.append(quality)
        
        # 2. 特征选择 - 基于质量分数和全局信息选择最优特征
        # 将所有特征拼接用于全局选择决策
        concat_feats = torch.cat(stripe_features, dim=1)  # [B, 4*C, H, W]
        selection_weights = self.selection_gate(concat_feats)  # [B, 4, 1, 1]
        
        # 3. 硬选择机制 - 选择权重最大的特征
        # 使用Gumbel-Softmax实现可微分的硬选择
        if self.training:
            # 训练时使用Gumbel-Softmax进行可微分选择
            gumbel_weights = F.gumbel_softmax(selection_weights.squeeze(-1).squeeze(-1), 
                                            tau=0.1, hard=True, dim=1)  # [B, 4]
            gumbel_weights = gumbel_weights.view(b, 4, 1, 1)
            selected_feat = sum(feat * weight for feat, weight in zip(stripe_features, 
                                                                    gumbel_weights.chunk(4, dim=1)))
        else:
            # 推理时使用argmax进行硬选择
            selected_idx = torch.argmax(selection_weights.squeeze(-1).squeeze(-1), dim=1)  # [B]
            selected_feat = torch.stack([stripe_features[idx][i] for i, idx in enumerate(selected_idx)])
        
        # 4. 特征增强
        # enhanced_feat = self.feature_enhance(selected_feat)
        
        return selected_feat


class CrossSourceFeatureInjection(nn.Module):
    """异源特征注入模块：基于缺陷区域的空间注意力进行特征注入"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 缺陷区域检测网络 - 从绝对相位特征中检测缺陷区域
        self.defect_detector = nn.Sequential(
            ConvBnLeakyRelu2d(channels, channels // 2),
            ConvBnLeakyRelu2d(channels // 2, channels // 4),
            nn.Conv2d(channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 空间注意力生成 - 生成缺陷区域的空间注意力权重
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # # 正弦特征适配
        # self.stripe_adapt = ConvBnLeakyRelu2d(channels, channels)
        
        # # 相位特征适配
        # self.phase_adapt = ConvBnLeakyRelu2d(channels, channels)
        
        # # 最终融合
        # self.final_fusion = ConvBnLeakyRelu2d(channels, channels)
        
    def forward(self, x):
        # phase_feat: 绝对相位特征 [B, C, H, W]
        # stripe_feat: 选中的正弦特征 [B, C, H, W]
        import cv2
        phase_feat, stripe_feat = x
        
        # 1. 缺陷区域检测 - 从绝对相位特征中检测缺陷
        # defect_mask = self.defect_detector(phase_feat)  # [B, 1, H, W]
        
        # 2. 空间注意力生成 - 理想情况下缺陷区域=1，其他区域=0
        spatial_attn = self.spatial_attention(stripe_feat)  # [B, 1, H, W]
        spatial_attn = spatial_attn
        
        # 3. 特征适配
        # adapted_stripe = self.stripe_adapt(stripe_feat)
        # adapted_phase = self.phase_adapt(phase_feat)
        
        # 4. 特征注入 - 按照您的设计思路
        # 缺陷区域：注入正弦特征 (spatial_attn ≈ 1)
        # 非缺陷区域：保持相位特征 (1 - spatial_attn ≈ 1)
        # injected_feat = adapted_stripe * spatial_attn + adapted_phase * (1 - spatial_attn)
        injected_feat = stripe_feat * spatial_attn + phase_feat * (1 - spatial_attn)
        
        # 5. 最终融合
        # output = self.final_fusion(injected_feat)
        
        return injected_feat, spatial_attn


class DefectAwareFusionModule(nn.Module):
    """缺陷感知特征融合模块 - 重新设计版本"""
    def __init__(self, c_all, channels, not_concat=[True, True], Cversion=1, stride=8):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.not_concat = not_concat
        n_img = c_all // channels
        C = [CrossSourceFeatureInjection]
        
        # 同源特征选择模块
        self.stripe_selection = StripeFeatureSelection(channels) if not_concat[0] else ChannelConcatReduction(channels*(n_img-1), channels)
        # print("stripe_selection: "+str(self.stripe_selection))
        
        # 异源特征注入模块ChannelConcatReduction(channels*2, channels)
        self.cross_injection = C[Cversion-1](channels) if not_concat[1] else ChannelConcatReduction(channels*2, channels)
        # print("cross_injection: "+str(self.cross_injection))
        
        # 输出适配层
        # self.output_conv = ConvBnLeakyRelu2d(channels, channels)
        
    def forward(self, feature_list):
        """
        Args:
            feature_list: list of 5 tensors
                - feature_list[0]: 绝对相位特征 [B, C, H, W]
                - feature_list[1:5]: 4个正弦条纹特征 [B, C, H, W] each
        Returns:
            fused_feature: 融合后的特征 [B, C, H, W]
        """
        # 1. 同源特征选择：从4个正弦特征中选择1个最优特征
        stripe_features = feature_list[1:5]  # 4个正弦特征
        selected_stripe = self.stripe_selection(stripe_features)
        
        # 2. 异源特征注入：基于缺陷区域进行特征注入
        phase_feature = feature_list[0]  # 绝对相位特征
        if self.not_concat[1]:
            fused_feature, spatial_attn = self.cross_injection([phase_feature, selected_stripe])
        else:
            fused_feature, spatial_attn = self.cross_injection([phase_feature, selected_stripe]), None
        
        # 3. 输出适配
        # output = self.output_conv(fused_feature)
        
        return [fused_feature, [spatial_attn, selected_stripe]]


class DefectAwareFusionModule_old(nn.Module):
    """缺陷感知特征融合模块"""
    def __init__(self, c_all, channels, stride=8):
        super().__init__()
        self.channels = channels
        self.stride = stride
        
        # 同源融合模块
        self.stripe_fusion = StripeFeatureFusion(channels)
        
        # 异源融合模块
        self.cross_fusion = CrossSourceFusion(channels)
        
        # 输出适配层
        self.output_conv = ConvBnLeakyRelu2d(channels, channels)
        
    def forward(self, feature_list):
        """
        Args:
            feature_list: list of 5 tensors
                - feature_list[0]: 绝对相位特征 [B, C, H, W]
                - feature_list[1:5]: 4个正弦条纹特征 [B, C, H, W] each
        Returns:
            fused_feature: 融合后的特征 [B, C, H, W]
        """
        # 1. 同源融合：融合4个正弦条纹特征
        stripe_features = feature_list[1:5]  # 4个正弦特征
        fused_stripe = self.stripe_fusion(stripe_features)
        
        # 2. 异源融合：融合绝对相位特征和正弦融合特征
        phase_feature = feature_list[0]  # 绝对相位特征
        fused_feature = self.cross_fusion(phase_feature, fused_stripe)
        
        # 3. 输出适配
        output = self.output_conv(fused_feature)
        
        return output


class IntraSourceFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        # self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.conv_fuse = nn.Conv2d(in_channels * 4, in_channels, 1)

    def forward(self, feats):  # feats: List of 4 feature maps (b,c,h,w)
        outs = []
        for x in feats:
            y = self.down(x)
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False) + x
            y = y * self.se(y)
            outs.append(y)
        fused = torch.cat(outs, dim=1)
        return self.conv_fuse(fused)


class CrossSourceFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv_aux = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, main_feat, aux_feat):  # main_feat: abs phase feat, aux_feat: fused sin feat
        aux = self.conv_aux(aux_feat)
        aux = aux * self.se(aux)
        return main_feat + aux


class DualStageFeatureFusion_old(nn.Module):
    def __init__(self, c_all, in_channels):
        super().__init__()
        self.isff = IntraSourceFusion(in_channels)
        self.csff = CrossSourceFusion(in_channels)

    def forward(self, x):  # abs_feat: (b,c,h,w), sin_feats: list of 4 (b,c,h,w)
        sin_fused = self.isff(x[1:])
        fused_feat = self.csff(x[0], sin_fused)
        return fused_feat


class DualStageFeatureFusion(nn.Module):
    """双阶段特征融合 - 重新设计版本"""
    def __init__(self, c_all, in_channels):
        super().__init__()
        # 第一阶段：同源特征选择
        self.stripe_selection = StripeFeatureSelection(in_channels)
        
        # 第二阶段：异源特征注入
        self.cross_injection = CrossSourceFeatureInjection(in_channels)

    def forward(self, x):
        # x[0]: 绝对相位特征 (b,c,h,w)
        # x[1:5]: 4个正弦特征 (b,c,h,w) each
        
        # 第一阶段：同源特征选择
        selected_stripe = self.stripe_selection(x[1:])
        
        # 第二阶段：异源特征注入
        fused_feat = self.cross_injection(x[0], selected_stripe)
        
        return fused_feat


class ChannelConcatReduction(nn.Module):
    """通道拼接降维模块：将输入的张量列表沿通道维度拼接后经过卷积降维到设定的通道数"""
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, padding=0, use_bn=True, use_activation=True):
        super().__init__()
        
        # 卷积降维层
        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.SiLU() if use_activation else nn.Identity()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.SiLU() if use_activation else nn.Identity()
            )
    
    def forward(self, feature_list):
        """
        Args:
            feature_list: list of tensors, each with shape [B, C_i, H, W]
        Returns:
            output: 降维后的特征 [B, output_channels, H, W]
        """
        # 沿通道维度拼接所有特征
        if len(feature_list) == 0:
            raise ValueError("输入的特征列表不能为空")
        
        # 检查所有特征的空间尺寸是否一致
        first_shape = feature_list[0].shape
        for i, feat in enumerate(feature_list):
            if feat.shape[2:] != first_shape[2:]:
                raise ValueError(f"特征 {i} 的空间尺寸 {feat.shape[2:]} 与第一个特征 {first_shape[2:]} 不一致")
        
        # 拼接特征
        concatenated = torch.cat(feature_list, dim=1)
        
        # 卷积降维
        output = self.conv(concatenated)
        
        return output


class CrossChannelAttention(nn.Module):
    """
    交叉通道注意力模块：基于CBAM思想，用于学习两个特征张量间的通道重要性
    
    核心思想：
    1. 利用两个特征的全局信息来生成通道注意力权重
    2. 通过交叉注意力让两个特征互相指导对方的通道重要性
    3. 生成自适应的通道融合权重
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 全局平均池化，提取通道全局信息
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 通道注意力生成网络 - 类似CBAM的通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 交叉注意力生成网络 - 利用两个特征的联合信息
        self.cross_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 融合权重生成网络
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 2, 1, bias=False),  # 输出2个权重
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat1, feat2):
        """
        Args:
            feat1, feat2: 两个输入特征张量 [B, C, H, W]
        Returns:
            融合后的特征张量 [B, C, H, W]
            注意力权重信息
        """
        B, C, H, W = feat1.shape
        
        # 1. 提取全局通道信息
        global1 = self.global_pool(feat1)  # [B, C, 1, 1]
        global2 = self.global_pool(feat2)  # [B, C, 1, 1]
        
        # 2. 生成各自的通道注意力权重（类似CBAM）
        attn1 = self.channel_attention(global1)  # [B, C, 1, 1]
        attn2 = self.channel_attention(global2)  # [B, C, 1, 1]
        
        # 3. 生成交叉注意力权重（两个特征的联合信息）
        global_concat = torch.cat([global1, global2], dim=1)  # [B, 2C, 1, 1]
        cross_attn = self.cross_attention(global_concat)  # [B, 2C, 1, 1]
        cross_attn1 = cross_attn[:, :C, :, :]  # [B, C, 1, 1]
        cross_attn2 = cross_attn[:, C:, :, :]  # [B, C, 1, 1]
        
        # 4. 融合自注意力和交叉注意力
        enhanced_attn1 = attn1 * cross_attn1  # 自身注意力 × 交叉注意力
        enhanced_attn2 = attn2 * cross_attn2
        
        # 5. 应用通道注意力
        enhanced_feat1 = feat1 * enhanced_attn1
        enhanced_feat2 = feat2 * enhanced_attn2
        
        # 6. 生成融合权重
        fusion_input = torch.cat([enhanced_feat1.mean(dim=(2,3), keepdim=True), 
                                enhanced_feat2.mean(dim=(2,3), keepdim=True)], dim=1)
        fusion_weights = self.fusion_weight(fusion_input)  # [B, 2, 1, 1]
        
        # 7. 加权融合
        fused_feat = (enhanced_feat1 * fusion_weights[:, 0:1, :, :] + 
                     enhanced_feat2 * fusion_weights[:, 1:2, :, :])
        
        # 返回融合特征和注意力信息
        attn_info = {
            'channel_attn1': enhanced_attn1,
            'channel_attn2': enhanced_attn2,
            'fusion_weights': fusion_weights
        }
        
        return fused_feat, attn_info


class CrossSpatialAttention(nn.Module):
    """
    交叉空间注意力模块：基于CBAM思想，用于学习两个特征张量间的空间重要性
    
    核心思想：
    1. 利用通道统计信息生成空间注意力（类似CBAM的空间注意力）
    2. 通过交叉注意力让两个特征互相指导对方的空间重要性
    3. 生成自适应的空间融合权重
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        
        # 空间注意力生成 - 类似CBAM的空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        
        # 交叉空间注意力生成
        self.cross_spatial_attention = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=kernel_size, padding=padding, bias=False),  # 输入4通道(2个特征各2个统计)
            nn.Sigmoid()
        )
        
        # 融合权重生成
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat1, feat2):
        """
        Args:
            feat1, feat2: 两个输入特征张量 [B, C, H, W]
        Returns:
            融合后的特征张量 [B, C, H, W]
            注意力权重信息
        """
        # 1. 提取通道统计信息（类似CBAM空间注意力）
        # 对每个特征计算通道维度的均值和最大值
        mean1 = torch.mean(feat1, dim=1, keepdim=True)  # [B, 1, H, W]
        max1, _ = torch.max(feat1, dim=1, keepdim=True)  # [B, 1, H, W]
        mean2 = torch.mean(feat2, dim=1, keepdim=True)  # [B, 1, H, W]
        max2, _ = torch.max(feat2, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 2. 生成各自的空间注意力权重（类似CBAM）
        spatial_input1 = torch.cat([mean1, max1], dim=1)  # [B, 2, H, W]
        spatial_input2 = torch.cat([mean2, max2], dim=1)  # [B, 2, H, W]
        
        spatial_attn1 = self.spatial_attention(spatial_input1)  # [B, 1, H, W]
        spatial_attn2 = self.spatial_attention(spatial_input2)  # [B, 1, H, W]
        
        # 3. 生成交叉空间注意力权重
        cross_input = torch.cat([mean1, max1, mean2, max2], dim=1)  # [B, 4, H, W]
        cross_spatial_attn = self.cross_spatial_attention(cross_input)  # [B, 2, H, W]
        cross_attn1 = cross_spatial_attn[:, 0:1, :, :]  # [B, 1, H, W]
        cross_attn2 = cross_spatial_attn[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 4. 融合自注意力和交叉注意力
        enhanced_spatial_attn1 = spatial_attn1 * cross_attn1
        enhanced_spatial_attn2 = spatial_attn2 * cross_attn2
        
        # 5. 应用空间注意力
        enhanced_feat1 = feat1 * enhanced_spatial_attn1
        enhanced_feat2 = feat2 * enhanced_spatial_attn2
        
        # 6. 生成融合权重
        fusion_input = torch.cat([enhanced_spatial_attn1, enhanced_spatial_attn2, 
                                cross_attn1, cross_attn2], dim=1)
        fusion_weights = self.fusion_weight(fusion_input)  # [B, 2, H, W]
        
        # 7. 加权融合
        fused_feat = (enhanced_feat1 * fusion_weights[:, 0:1, :, :] + 
                     enhanced_feat2 * fusion_weights[:, 1:2, :, :])
        
        # 返回融合特征和注意力信息
        attn_info = {
            'spatial_attn1': enhanced_spatial_attn1,
            'spatial_attn2': enhanced_spatial_attn2,
            'fusion_weights': fusion_weights
        }
        
        return fused_feat, attn_info

