"""A PyTorch copy of any engine model.

    net = twin.to_torch(model, device="cuda")       # MLP, DeepMitsuNet, MitsuConvNet, MitsuGPT, MitsuLM, or your own stack

`to_torch` walks the engine model and builds the same structure from PyTorch modules,
with the same attribute names, so every parameter has the same name on both sides. It
then copies the weights. The result carries `engine_layout`: the engine's parameter order
with, for each parameter, whether its layout is transposed and whether weight decay
applies. That list is what the optimiser, the checkpoint code and the parity check use.

Every layer in the engine has a mirror here. A layer of your own (including an
AutoModule) needs one line to say what its PyTorch form is: see `register`.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .. import activations as A, blocks as B, conv as C, layers as L, models as M, module as base
from .functional import _work
from .modules import MITSU, RMSNorm, Attention, TiedHead

_REGISTRY = {}


def register(engine_class, builder):
    """Tell `to_torch` how to mirror a layer of your own.

    builder(engine_module, convert) returns a torch.nn.Module. `convert` is to_torch's
    converter for sub-modules. The PyTorch module must give its parameters the same
    attribute names as the engine layer, in the same shapes (weights are copied as they
    are; only the engine's own Linear is transposed)."""
    _REGISTRY[engine_class] = builder


# ---- mirrors of the engine's layers --------------------------------------------------

class Scaled(nn.Module):
    """gain * fn(x): the engine's Activation."""

    def __init__(self, fn, gain=1.0):
        super().__init__()
        self.fn, self.gain = fn, float(gain)

    def forward(self, x):
        y = self.fn(x)
        return y if self.gain == 1.0 else y * self.gain


class LayerNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.gain, self.bias = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, (self.dim,), self.gain, self.bias, self.eps)


class GroupNorm(nn.Module):
    def __init__(self, groups, channels, eps):
        super().__init__()
        self.groups, self.eps = groups, eps
        self.gain, self.bias = nn.Parameter(torch.ones(channels)), nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return F.group_norm(x, self.groups, self.gain, self.bias, self.eps)


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


class Sequential(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class FeedForward(nn.Module):
    def __init__(self, fc, act, out):
        super().__init__()
        self.fc, self.act, self.out = fc, act, out

    def forward(self, x):
        return self.out(self.act(self.fc(x)))


class GLU(nn.Module):
    """The engine's MitsuGLU in full: optional bias and any seat constant c."""

    def __init__(self, w1, w3, gate, w2):
        super().__init__()
        self.w1, self.w3, self.gate, self.w2 = w1, w3, gate, w2

    def forward(self, x):
        return self.w2(self.gate(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, norm1, attn, drop1, norm2, ffn, drop2):
        super().__init__()
        self.norm1, self.attn, self.drop1, self.norm2, self.ffn, self.drop2 = norm1, attn, drop1, norm2, ffn, drop2

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        return x + self.drop2(self.ffn(self.norm2(x)))


class DeepMitsuNet(nn.Module):
    def __init__(self, stem, blocks, norm, head):
        super().__init__()
        self.stem, self.blocks, self.norm, self.head = stem, nn.ModuleList(blocks), norm, head

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


class MitsuGPT(nn.Module):
    def __init__(self, tok, pos, drop, blocks, norm, head, context):
        super().__init__()
        self.context = context
        self.tok, self.pos, self.drop, self.blocks, self.norm, self.head = tok, nn.Parameter(pos), drop, nn.ModuleList(blocks), norm, head

    def forward(self, idx):
        if idx.shape[1] > self.context:
            raise ValueError(f"sequence length {idx.shape[1]} exceeds the context of {self.context}")
        x = self.drop(self.tok(idx) + self.pos[: idx.shape[1]])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


# ---- the converter -------------------------------------------------------------------

def _linear(m):
    return nn.Linear(m.d_in, m.d_out, bias=m.bias is not None)


def _activation(m):
    if isinstance(m, A.MITSU):
        return MITSU(m.c, m.gain)
    fn = {A.ReLU: nn.ReLU(), A.GELU: nn.GELU(approximate="tanh"), A.SiLU: nn.SiLU(), A.Tanh: nn.Tanh()}.get(type(m))
    if fn is None:
        raise TypeError(f"no PyTorch form is known for the activation {type(m).__name__}; add one with twin.register")
    return Scaled(fn, m.gain)


def _convert(m, embeddings):
    """The PyTorch mirror of one engine module. `embeddings` maps an engine Embedding's
    Parameter to its mirror, so a TiedHead can share the same table."""
    go = lambda child: _convert(child, embeddings)
    for cls, builder in _REGISTRY.items():
        if isinstance(m, cls):
            return builder(m, go)
    if isinstance(m, M.MitsuLM):
        from .model import TorchMitsuLM
        return TorchMitsuLM(m.config, init=False)
    if isinstance(m, M.MitsuGPT):
        return MitsuGPT(go(m.tok), torch.zeros(m.pos.data.shape), go(m.drop), [go(b) for b in m.blocks], go(m.norm), go(m.head), m.context)
    if isinstance(m, M.DeepMitsuNet):
        return DeepMitsuNet(go(m.stem), [go(b) for b in m.blocks], go(m.norm), go(m.head))
    if isinstance(m, B.TransformerBlock):
        return TransformerBlock(go(m.norm1), go(m.attn), go(m.drop1), go(m.norm2), go(m.ffn), go(m.drop2))
    if isinstance(m, B.MultiHeadSelfAttention):
        return Attention(m.dim, m.n_heads, m.rope_base, causal=m.causal, rope=m.rope, bias=m.qkv.bias is not None)
    if isinstance(m, B.MitsuGLU):
        return GLU(go(m.w1), go(m.w3), go(m.gate), go(m.w2))
    if isinstance(m, B.FeedForward):
        return FeedForward(go(m.fc), go(m.act), go(m.out))
    if isinstance(m, base.Sequential):                           # also MLP and MitsuConvNet
        return Sequential([go(layer) for layer in m.layers])
    if isinstance(m, base.Residual):
        return Residual(go(m.fn))
    if isinstance(m, base.Identity):
        return nn.Identity()
    if isinstance(m, L.Linear):
        return _linear(m)
    if isinstance(m, L.Embedding):
        mirror = nn.Embedding(m.num_embeddings, m.dim)
        embeddings[id(m.weight)] = mirror
        return mirror
    if isinstance(m, L.TiedHead):
        if id(m.weight) not in embeddings:
            raise TypeError("a TiedHead must come after the Embedding it shares its table with")
        return TiedHead(embeddings[id(m.weight)])
    if isinstance(m, L.LayerNorm):
        return LayerNorm(m.dim, m.eps)
    if isinstance(m, L.RMSNorm):
        return RMSNorm(m.dim, m.eps)
    if isinstance(m, L.Dropout):
        return nn.Dropout(m.p)
    if isinstance(m, L.Flatten):
        return nn.Flatten(1)
    if isinstance(m, C.Conv2d):
        return nn.Conv2d(m.c_in, m.c_out, m.kernel, m.stride, m.padding, bias=m.bias is not None)
    if isinstance(m, C.MaxPool2d):
        return nn.MaxPool2d(m.kernel, m.stride)
    if isinstance(m, C.AvgPool2d):
        return nn.AvgPool2d(m.kernel, m.stride)
    if isinstance(m, C.GlobalAvgPool2d):
        return GlobalAvgPool2d()
    if isinstance(m, C.GroupNorm):
        return GroupNorm(m.groups, m.channels, m.eps)
    if isinstance(m, (A.MITSU, A.Activation)):
        return _activation(m)
    raise TypeError(f"no PyTorch form is known for {type(m).__name__}. Layers of your own (including AutoModule subclasses) need "
                    f"twin.register({type(m).__name__}, builder); see mitsu_engine/twin/mirror.py")


def _owned_parameters(module, prefix="", seen=None):
    """(name, Parameter, owning module) in the engine's parameter order; a shared
    Parameter appears once, under its first owner (as Module.named_parameters does)."""
    seen = set() if seen is None else seen
    for name, value in module.__dict__.items():
        if isinstance(value, base.Parameter) and id(value) not in seen:
            seen.add(id(value))
            yield prefix + name, value, module
    for name, child in module.children():
        yield from _owned_parameters(child, prefix + name + ".", seen)


def build_layout(model, net):
    """The engine's parameter list for `net`: [(name, torch parameter, transposed, decay)]."""
    own = dict(net.named_parameters())
    layout, missing = [], []
    for name, p, owner in _owned_parameters(model):
        if name not in own:
            missing.append(name)
            continue
        transposed = isinstance(owner, L.Linear) and p is owner.weight
        shape = p.data.shape[::-1] if transposed else p.data.shape
        if tuple(own[name].shape) != shape:
            raise ValueError(f"{name}: the engine has shape {shape} (in PyTorch layout), the PyTorch module {tuple(own[name].shape)}")
        layout.append((name, own[name], transposed, bool(p.decay)))
    unexpected = [n for n in own if n not in {entry[0] for entry in layout}]
    if missing or unexpected:
        raise KeyError(f"the PyTorch module's parameters do not match the engine's. missing: {missing[:5]}  unexpected: {unexpected[:5]}")
    return layout


def to_torch(model, device=None, dtype=None):
    """A PyTorch copy of an engine model, holding the same weights.

    device   where to put it ("cuda", "cpu", a torch.device); None leaves it on the CPU
    dtype    torch.float32 or torch.float64; None keeps the engine model's own precision

    The result is an ordinary torch.nn.Module with one extra attribute, `engine_layout`.
    Train it with twin.TorchTrainer and twin.EngineAdamW; copy weights back with
    twin.to_engine; compare it with the original with twin.compare_models."""
    from .convert import load_engine_state
    net = _convert(model, {})
    params = [p for _, p in model.named_parameters()]
    if dtype is None:
        dtype = torch.float64 if params and params[0].data.dtype == np.float64 else torch.float32
    net.to(device=device, dtype=dtype)
    if not hasattr(net, "engine_layout"):                         # TorchMitsuLM computes its own
        net.engine_layout = build_layout(model, net)
    load_engine_state(net, model.state_dict())
    net.train(model.training)
    return net


def to_engine(net, model):
    """Copy the PyTorch module's weights into the engine model it mirrors. Returns `model`."""
    from .convert import engine_state
    return model.load_state_dict(engine_state(net))
