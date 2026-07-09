import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import math
from basicsr.models.archs.scanutils import index_select_2d

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class MSFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(MSFeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))
        x = torch.cat([x1, x2], dim=1)
        x = self.project_out(x)
        return x

class MambaEngine(nn.Module):
    r""" 
        Input:  xs | xs.shape  -->  (b, k, d, l).
        Output: ys | ys.shape  -->  (b, k, d, l).
    """    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_inner=0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner

        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.selective_scan = selective_scan_fn
        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.conv_mask = SpatialAttention()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, xs, h, w): 
        b, k, d, l = xs.shape
        # 
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight) # 矩阵乘法
        # b k c l 
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)

        Bs = rearrange(Bs, 'b k d (h1 w1) -> (b k) d h1 w1', h1=h, w1=w)
        sa = self.conv_mask(Bs)
        Bs =  Bs * sa + Bs 
        Bs = rearrange(Bs, '(b k) d h1 w1 -> b k d (h1 w1)', b=b, k=k)

        xs = xs.float().view(b, -1, l) 
        dts = dts.contiguous().float().view(b, -1, l) 
        Bs = Bs.float().view(b, k, -1, l)
        Cs = Cs.float().view(b, k, -1, l) 
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_ys = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)
        
        assert out_ys.dtype == torch.float
        
        return out_ys

class ScanPattern(nn.Module):
    r""" 
        Input:  x,    x.shape  -->  (b, d, h, w).
        Output: xs,  xs.shape  -->  (b, k, d, l).
    """ 
    def __init__(
        self, 
        s2s_engine=None,
        route_dict=None, 
        **kwargs
    ):
        super().__init__()

        self.scan_route_dict = route_dict
        self.seq2seq_engine  = s2s_engine 

        self.scan_tind_0 = self.scan_route_dict[0, 0]
        self.scan_tind_1 = self.scan_route_dict[1, 0]
        self.scan_pind_0 = self.scan_route_dict[0, 1]
        self.scan_pind_1 = self.scan_route_dict[1, 1]

    def ScanRoutes(self, x):    
        b, d, h, w = x.shape
        k = 4
        l = h*w
        x_scan1d_0     = index_select_2d(x, self.scan_tind_0)
        x_scan1d_1     = index_select_2d(x, self.scan_tind_1)
        x_scan1d_0_inv = torch.flip(x_scan1d_0, dims=[-1])
        x_scan1d_1_inv = torch.flip(x_scan1d_1, dims=[-1])
        xs = torch.stack([x_scan1d_0, x_scan1d_0_inv, x_scan1d_1, x_scan1d_1_inv], dim=1).view(b, k, -1, l)
        return xs
    
    def ReArrange(self, ys_s2s):
        b, k, d, l  = ys_s2s.shape
        y_re_0      = torch.index_select(ys_s2s[:,0], dim=-1, index=self.scan_pind_0)
        y_re_1      = torch.index_select(ys_s2s[:,2], dim=-1, index=self.scan_pind_1)        
        y_re_0_inv  = torch.index_select(torch.flip(ys_s2s[:,1], dims=[-1]), dim=-1, index=self.scan_pind_0)
        y_re_1_inv  = torch.index_select(torch.flip(ys_s2s[:,3], dims=[-1]), dim=-1, index=self.scan_pind_1)

        return y_re_0, y_re_0_inv, y_re_1, y_re_1_inv

    def forward(self, x):
        b, d, h, w = x.shape
        xs     = self.ScanRoutes(x)
        ys_s2s = self.seq2seq_engine(xs, h, w)
        y1, y2, y3, y4 = self.ReArrange(ys_s2s)
        assert y1.dtype == torch.float32
        return y1, y2, y3, y4

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            route_dict=None, 
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.route_dict = route_dict

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, 
                                out_channels=self.d_inner, 
                                groups=self.d_inner, 
                                bias=conv_bias, 
                                kernel_size=d_conv, 
                                padding=(d_conv - 1) // 2, 
                                **factory_kwargs)
        self.act = nn.SiLU()


        self.mambaengine = MambaEngine(self.d_model, 
                                        d_state, 
                                        self.d_inner, 
                                        dt_rank, 
                                        dt_min, 
                                        dt_max, 
                                        dt_init, 
                                        dt_scale, 
                                        dt_init_floor, 
                                        bias, 
                                        device, 
                                        dtype)
        
        self.scanning = ScanPattern(s2s_engine=self.mambaengine,  route_dict=self.route_dict)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    def forward(self, x, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.scanning(x)
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
