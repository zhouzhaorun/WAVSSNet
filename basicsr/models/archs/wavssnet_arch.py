import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.mamba_blocks import SS2D, MSFeedForward
from basicsr.models.archs.wave_tf import HaarDownsampling
from basicsr.models.archs.modules import LayerNorm, OverlapPatchEmbed

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )
    def forward(self, x):
        return self.cab(x)

class HybridAttBlock(nn.Module):
    def __init__(self, dim, route_dict):
        super(HybridAttBlock, self).__init__()
        self.mamba = SS2D(dim, route_dict=route_dict)
        self.channel_attention = CAB(num_feat=dim, compress_ratio=4)

    def forward(self, x):
        att1 = self.mamba(x.permute(0,2,3,1)).permute(0,3,1,2)
        att2 = self.channel_attention(x)
        return att1 + att2

class TransformerBlock(nn.Module):
    def __init__(self, dim, route_dict, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HybridAttBlock(dim, route_dict=route_dict)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MSFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class TransformerLayer(nn.Module):
    def __init__(self, depth, dim, route_dict, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerLayer, self).__init__()
        self.depth = depth
        self.blocks_connected = nn.ModuleDict()
        for block_i in range(depth):
            self.blocks_connected[f'block{block_i}'] = TransformerBlock(dim=dim, 
                                                                        route_dict=route_dict,
                                                                        ffn_expansion_factor=ffn_expansion_factor, 
                                                                        bias=bias, 
                                                                        LayerNorm_type=LayerNorm_type)
    def forward(self, x):
        block_feat = x
        for block_i in range(self.depth):
            block = self.blocks_connected[f'block{block_i}']
            block_feat = block(block_feat)
        return block_feat


class wavssnet(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        ffn_expansion_factor = 2.66,
        route_dict_path=None,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):
        super(wavssnet, self).__init__()
        self.route_dict = torch.load(route_dict_path)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.wave = HaarDownsampling(dim)
        self.x1_wave_conv1 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)
        self.x1_wave_conv2 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)

        self.x2_wave_conv1 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)
        self.x2_wave_conv2 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)

        self.x3_wave_conv1 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)
        self.x3_wave_conv2 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)

        self.act = nn.SiLU()

        self.ll_branch1 = TransformerLayer(num_blocks[1], 
                                                dim=dim, 
                                                route_dict=self.route_dict[1]['layer_'+str(1)+'.cxsz_tpinds'].to('cuda'), 
                                                ffn_expansion_factor=ffn_expansion_factor, 
                                                bias=bias, 
                                                LayerNorm_type=LayerNorm_type)
        
        self.ll_branch2 = TransformerLayer(num_blocks[2], 
                                                dim=dim, 
                                                route_dict=self.route_dict[2]['layer_'+str(2)+'.cxsz_tpinds'].to('cuda'), 
                                                ffn_expansion_factor=ffn_expansion_factor, 
                                                bias=bias, 
                                                LayerNorm_type=LayerNorm_type)
        
        self.ll_branch3 = TransformerLayer(num_blocks[3], 
                                                dim=dim, 
                                                route_dict=self.route_dict[3]['layer_'+str(3)+'.cxsz_tpinds'].to('cuda'), 
                                                ffn_expansion_factor=ffn_expansion_factor, 
                                                bias=bias, 
                                                LayerNorm_type=LayerNorm_type)


        self.refinement = TransformerLayer(num_blocks[0], 
                                                dim=dim, 
                                                route_dict=self.route_dict[0]['layer_'+str(0)+'.cxsz_tpinds'].to('cuda'), 
                                                ffn_expansion_factor=ffn_expansion_factor, 
                                                bias=bias, 
                                                LayerNorm_type=LayerNorm_type)
        

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)#*2**1

    def forward(self, inp_img, mask): 
        comp_img = torch.cat((inp_img, mask), dim=1)
        feat = self.patch_embed(comp_img)

        x_l1, x_h1 = self.wave(feat) 
        x_l2, x_h2 = self.wave(x_l1)
        x_l3, x_h3 = self.wave(x_l2)

        # h1 h2 h3
        x_h1 = self.x1_wave_conv2(self.act(self.x1_wave_conv1(x_h1)))
        x_h2 = self.x2_wave_conv2(self.act(self.x2_wave_conv1(x_h2)))
        x_h3 = self.x3_wave_conv2(self.act(self.x3_wave_conv1(x_h3)))        

        # l3
        x_l3_out = self.ll_branch3(x_l3)

        x_l3_out = x_l3 + x_l3_out
        up4_feat = self.wave(torch.cat([x_l3_out, x_h3], dim=1), rev=True)
        x_l2 = x_l2 + up4_feat

        # l2
        x_l2_out = self.ll_branch2(x_l2)

        x_l2_out = x_l2_out + x_l2
        up2_feat = self.wave(torch.cat([x_l2_out, x_h2], dim=1), rev=True)
        x_l1 = x_l1 + up2_feat

        # l1
        x_l1_out = self.ll_branch1(x_l1)

        x_l1_out = x_l1_out + x_l1
        up1_feat = self.wave(torch.cat([x_l1_out, x_h1], dim=1), rev=True)

        # refinement
        ref_out = self.refinement(up1_feat)
        ref_out = ref_out + up1_feat
        out_feat = self.output(ref_out) + inp_img
        return out_feat

