"""mitsu_engine — a neural-network engine built around the MITSU activation.

numpy only. Every layer has a hand-written backward pass, checked against finite
differences. See README.md for a tour.

    import mitsu_engine as me
    model = me.DeepMitsuNet(d_in=784, d_out=10, width=256, depth=16)
    trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=1e-3))
    trainer.fit(me.DataLoader(X, y, batch_size=128), epochs=5)
"""
from .config import set_dtype, get_dtype, manual_seed, get_rng, get_rng_state, set_rng_state, no_grad, grad_enabled
from . import functional, init
from .module import Parameter, Module, Sequential, Residual, Identity
from .layers import Linear, Embedding, TiedHead, LayerNorm, RMSNorm, Dropout, Flatten
from .activations import MITSU, ReLU, GELU, SiLU, Tanh, Activation, get_activation
from .blocks import FeedForward, MitsuGLU, MultiHeadSelfAttention, TransformerBlock, mitsu_block, residual_branch
from .models import MLP, DeepMitsuNet, MitsuGPT, MitsuLM, LMConfig, sample_token
from .losses import CrossEntropy, MSE, accuracy
from .optim import SGD, Adam, AdamW, clip_grad_norm, grad_norm, ConstantLR, WarmupCosine, StepLR
from .data import DataLoader, train_val_split, Standardizer, CharTokenizer, TextBatcher, make_spirals
from .datasets import load_mnist, load_text
from .io import save_checkpoint, load_checkpoint, read_meta
from .gradcheck import gradcheck
from .trainer import Trainer, evaluate_lm
from .tokenizer import BPETokenizer
from .corpus import prepare_corpus, load_corpus, open_tokens, write_tokens, encode_file, iter_text, token_dtype

__version__ = "0.2.0"
