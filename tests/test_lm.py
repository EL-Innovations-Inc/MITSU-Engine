"""The language model: rotary positions, tied embeddings, cached generation, persistence."""
import numpy as np
import pytest
import mitsu_engine as me
from mitsu_engine.blocks import rope_tables, rope_rotate, rope_unrotate


def small_lm(**kw):
    args = dict(context=12, d_model=16, n_layers=3, n_heads=4)
    args.update(kw)
    return me.MitsuLM(17, **args)


# ---- rotary positions ---------------------------------------------------------

def test_rotation_preserves_length_and_unrotate_is_its_transpose():
    r = np.random.default_rng(0)
    x, g = r.standard_normal((2, 3, 7, 8)), r.standard_normal((2, 3, 7, 8))
    cos, sin = rope_tables(7, 8, dtype=np.float64)
    assert np.allclose((rope_rotate(x, cos, sin) ** 2).sum(-1), (x ** 2).sum(-1))
    assert np.allclose(rope_unrotate(rope_rotate(x, cos, sin), cos, sin), x)
    assert np.isclose((rope_rotate(x, cos, sin) * g).sum(), (x * rope_unrotate(g, cos, sin)).sum())


def test_rotary_attention_depends_only_on_relative_position():
    me.set_dtype(np.float64)
    att = me.MultiHeadSelfAttention(8, 2, rope=True)
    x = np.random.default_rng(0).standard_normal((1, 6, 8))
    q0, k0, _, _ = att._qkv(x, 0)
    q9, k9, _, _ = att._qkv(x, 9)
    assert np.allclose(q0 @ k0.transpose(0, 1, 3, 2), q9 @ k9.transpose(0, 1, 3, 2))
    assert not np.allclose(q0, q9)


def test_rotary_needs_even_head_size():
    with pytest.raises(ValueError):
        me.MultiHeadSelfAttention(6, 2, rope=True)
    with pytest.raises(ValueError):
        me.MitsuLM(10, d_model=12, n_heads=4)


# ---- tied embeddings ------------------------------------------------------------

def test_tied_table_is_one_parameter():
    tied, untied = small_lm(), small_lm(tie_embeddings=False)
    params = tied.parameters()
    assert len(params) == len({id(p) for p in params})
    names = [n for n, _ in tied.named_parameters()]
    assert "tok.weight" in names and "head.weight" not in names
    assert "head.weight" in [n for n, _ in untied.named_parameters()]
    assert untied.num_parameters() - tied.num_parameters() == 17 * 16
    assert tied.head.weight is tied.tok.weight
    assert len(me.AdamW(tied.parameters()).params) == len(names)


def test_parameter_count_formula():
    lm = me.MitsuLM(100, context=8, d_model=24, n_layers=3, n_heads=4, ffn_hidden=64)
    per_block = 24 + 24 * 72 + 24 * 24 + 24 + 3 * 24 * 64
    assert lm.num_parameters() == 100 * 24 + 3 * per_block + 24
    assert me.MitsuLM(100, d_model=24, n_heads=4).blocks[0].ffn.hidden == 64


# ---- cached generation ----------------------------------------------------------

def test_incremental_step_matches_full_forward():
    me.set_dtype(np.float64)
    lm = small_lm().eval()
    idx = np.random.default_rng(0).integers(0, 17, (2, 12))
    with me.no_grad():
        full = lm(idx)
    logits, caches = lm.step(idx[:, :5])
    parts = [logits]
    for t in range(5, 12):
        logits, caches = lm.step(idx[:, t:t + 1], caches)
        parts.append(logits)
    assert np.abs(np.concatenate(parts, 1) - full).max() < 1e-10
    assert caches[0]["k"].shape == (2, 4, 12, 4)


@pytest.mark.parametrize("prompt_len,new", [(4, 30), (12, 5), (30, 9), (1, 40)])
def test_cached_generation_equals_recomputing(prompt_len, new):
    lm = small_lm()
    prompt = np.random.default_rng(prompt_len).integers(0, 17, (2, prompt_len))
    greedy = lm.generate(prompt, new, temperature=0)
    assert greedy.shape == (2, prompt_len + new)
    assert np.array_equal(greedy, lm.generate(prompt, new, temperature=0, use_cache=False))
    a = lm.generate(prompt, new, temperature=0.9, top_k=5, top_p=0.9, rng=3)
    b = lm.generate(prompt, new, temperature=0.9, top_k=5, top_p=0.9, rng=3, use_cache=False)
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() < 17


def test_generate_keeps_training_mode_and_stops_on_stop_token():
    lm = small_lm(dropout=0.1)
    assert lm.training
    out = lm.generate([1, 2, 3], 5, temperature=0)
    assert lm.training and out.shape == (1, 8) and np.array_equal(out[0, :3], [1, 2, 3])
    first = int(lm.generate([1, 2, 3], 1, temperature=0)[0, -1])
    assert lm.generate([1, 2, 3], 20, temperature=0, stop_token=first).shape == (1, 4)


def test_sample_token():
    logits = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    assert me.sample_token(logits, temperature=0)[0] == 0
    r = np.random.default_rng(0)
    draws = np.array([me.sample_token(logits, 1.0, rng=r)[0] for _ in range(4000)])
    assert np.abs(np.bincount(draws, minlength=4) / 4000 - [0.5, 0.3, 0.15, 0.05]).max() < 0.03
    assert set(np.array([me.sample_token(logits, 1.0, top_k=2, rng=r)[0] for _ in range(300)])) == {0, 1}
    assert set(np.array([me.sample_token(logits, 1.0, top_p=0.6, rng=r)[0] for _ in range(300)])) == {0, 1}
    assert set(np.array([me.sample_token(logits, 1.0, top_p=0.4, rng=r)[0] for _ in range(100)])) == {0}


# ---- persistence ----------------------------------------------------------------

def test_config_round_trip_and_model_save_load(tmp_path):
    cfg = me.LMConfig(vocab_size=17, context=12, d_model=16, n_layers=2, n_heads=4, dropout=0.1, tie_embeddings=False)
    assert me.LMConfig.from_dict({**cfg.to_dict(), "future_field": 1}) == cfg
    lm = me.MitsuLM(cfg, rng=1).eval()
    path = str(tmp_path / "model.npz")
    lm.save(path, note="hello")
    again = me.MitsuLM.load(path).eval()
    idx = np.random.default_rng(0).integers(0, 17, (2, 12))
    assert again.config == cfg and np.array_equal(lm(idx), again(idx))
    assert me.read_meta(path)["note"] == "hello"
    plain = str(tmp_path / "plain.npz")
    me.save_checkpoint(plain, me.Linear(2, 2))
    with pytest.raises(ValueError):
        me.MitsuLM.load(plain)


def test_context_limit():
    with pytest.raises(ValueError):
        small_lm()(np.zeros((1, 13), dtype=np.int64))


# ---- it learns --------------------------------------------------------------------

def test_lm_memorises_a_pattern_and_regenerates_it_with_the_cache():
    pattern = np.array([3, 1, 4, 1, 5, 9, 2, 6] * 40)
    lm = me.MitsuLM(10, context=16, d_model=32, n_layers=2, n_heads=4, rng=0)
    batcher = me.TextBatcher(pattern, context=16, batch_size=16, rng=1)
    trainer = me.Trainer(lm, me.AdamW(lm.parameters(), lr=3e-3), clip_norm=1.0, metric=None, log=None)
    trainer.fit_steps(batcher, 200)
    assert me.evaluate_lm(lm, batcher.windows(8))["loss"] < 0.15
    out = lm.generate([3, 1, 4, 1, 5, 9, 2, 6], 24, temperature=0)[0]
    assert out.tolist() == [3, 1, 4, 1, 5, 9, 2, 6] * 4
