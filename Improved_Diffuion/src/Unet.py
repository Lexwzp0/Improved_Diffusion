import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
#%%
def exists(x):
    return x is not None
#%%
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def sinusoidal_embedding(t, dim):
    """
    生成正弦位置编码
    :param t: 时间步张量 [batch_size,]
    :param dim: 嵌入维度
    :return: 嵌入向量 [batch_size, dim]
    """
    device = t.device
    half_dim = dim // 2

    # 修复点1：将数值转换为Tensor，并指定device
    emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)  # 注意使用10000.0确保浮点数

    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t.float()[:, None] * emb[None, :]  # [batch_size, half_dim]

    # 拼接正弦和余弦分量
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [batch_size, dim]

    # 如果dim是奇数，补零
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=1)

    return emb


#%%
class ConditionalEmbedding(nn.Module):
    def __init__(self, num_classes, time_dim=256, label_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim*4)
        )
        self.label_embed = nn.Embedding(num_classes, label_dim)
        self.fusion = nn.Sequential(
            nn.Linear(time_dim*4 + label_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, time_dim)
        )

    def forward(self, t, y):
        # t: [B,] 时间步
        # y: [B,] 标签
        t_emb = sinusoidal_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)  # [B, time_dim]

        l_emb = self.label_embed(y).squeeze(1)     # [B, label_dim]

        # 融合时间与标签信息
        combined = torch.cat([t_emb, l_emb], dim=1)
        return self.fusion(combined)    # [B, time_dim]
#%%
class MySequential(nn.Sequential):
    def forward(self, x, t_emb):
        for module in self:
            if isinstance(module, ConditionalBlock):  # 仅对特定模块传参
                x = module(x, t_emb)
            else:  # 其他模块按默认方式处理
                x = module(x)
        return x
#%%


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b"
                     " h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
#%%
from einops import rearrange

class ConditionalBlock(nn.Module):
    """基于你原有MyBlock改造的条件版本"""
    def __init__(self, in_ch, out_ch, cond_dim, mult=1):
        """
        Args:
            cond_dim: 条件向量的维度 (time+label的融合维度)
        """
        super().__init__()

        # 修改后的条件投影层（移除偏置项）验证条件注入的有效性
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, out_ch*2, bias=False),  # 关键修改：bias=False
            nn.GELU()
        )

        # 保持原有卷积结构
        self.ds_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.conv = nn.Sequential(
            nn.GroupNorm(1, out_ch),
            nn.Conv2d(out_ch, out_ch * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_ch * mult),
            nn.Conv2d(out_ch * mult, out_ch, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond_emb):
        """输入变化：t_emb → cond_emb (融合时间+标签的条件向量)"""
        h = self.ds_conv(x)

        # 条件注入 (scale and shift)
        scale, shift = self.cond_mlp(cond_emb).chunk(2, dim=1)  # [B, 2*out_ch] → [B, out_ch], [B, out_ch]
        h = h * (1 + scale[:, :, None, None])  # 缩放
        h = h + shift[:, :, None, None]        # 偏移

        h = self.conv(h)
        return h + self.res_conv(x)  # 保持原有残差连接

#%%
# 残差模块，将输入加到输出上
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
#%%
class ConditionalDiffusionUNet(nn.Module):
    def __init__(self, num_classes, time_dim=128, label_dim=64):
        super().__init__()
        chs = [1, 64, 128, 256]

        # 替换为条件嵌入层
        self.cond_embed = ConditionalEmbedding(
            num_classes=num_classes,
            time_dim=time_dim,
            label_dim=label_dim
        )
        cond_dim = time_dim  # 条件向量的总维度

        # 下采样路径（修改所有MyBlock的cond_dim）
        self.down = nn.ModuleList([
            MySequential(
                ConditionalBlock(chs[i], chs[i+1], cond_dim=cond_dim),
                ConditionalBlock(chs[i+1], chs[i+1], cond_dim=cond_dim),
                Residual(PreNorm(chs[i+1], LinearAttention(chs[i+1])))
            ) for i in range(len(chs)-1)
        ])

        # 中间层
        self.mid = MySequential(
            ConditionalBlock(chs[-1], chs[-1], cond_dim=cond_dim),
            Residual(PreNorm(chs[-1], Attention(chs[-1]))),
            ConditionalBlock(chs[-1], chs[-1], cond_dim=cond_dim)
        )

        # 上采样路径
        self.up = nn.ModuleList([
            MySequential(
                ConditionalBlock(chs[i+1]*2, chs[i], cond_dim=cond_dim),
                ConditionalBlock(chs[i], chs[i], cond_dim=cond_dim),
                Residual(PreNorm(chs[i], LinearAttention(chs[i])))
            ) for i in reversed(range(len(chs)-1))
        ])

        self.final = nn.Conv2d(chs[0], 2, 1)

    def forward(self, x, t, y):
        """新增标签y作为输入"""
        cond_emb = self.cond_embed(t, y)  # 获取融合条件向量
        skips = []

        # 编码器（传递cond_emb）
        for block in self.down:
            x = block(x, cond_emb)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=(2,1))

        # 中间处理
        x = self.mid(x, cond_emb)

        # 解码器
        for i, block in enumerate(self.up):
            x = F.interpolate(x, scale_factor=(2,1), mode='nearest')
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = block(x, cond_emb)

        return self.final(x)