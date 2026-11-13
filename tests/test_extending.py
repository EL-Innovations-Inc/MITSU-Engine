"""The 'write your own layer' example from the README, kept honest by running it."""
import numpy as np
import mitsu_engine as me


class Scale(me.Module):
    """y = x * exp(log_scale): one learned positive gain per feature."""

    def __init__(self, dim):
        self.log_scale = me.Parameter(np.zeros(dim), decay=False)
        self._c_x = None

    def forward(self, x):
        if me.grad_enabled():
            self._c_x = x
        return x * np.exp(self.log_scale.data)

    def backward(self, grad):
        s = np.exp(self.log_scale.data)
        self.log_scale.grad += (grad * self._c_x * s).reshape(-1, s.size).sum(0)
        return grad * s


def test_custom_layer_passes_gradcheck_and_trains():
    me.set_dtype(np.float64)
    x = np.random.default_rng(0).standard_normal((4, 6))
    model = me.Sequential(me.Linear(6, 6), Scale(6), me.MITSU(), me.Linear(6, 2))
    err, where = me.gradcheck(model, x)
    assert err < 1e-5, where
    assert any(name.endswith("log_scale") for name, _ in model.named_parameters())
    opt, y = me.Adam(model.parameters(), lr=0.05), np.array([0, 1, 0, 1])
    first = last = None
    for _ in range(100):
        opt.zero_grad()
        loss, g = me.CrossEntropy()(model(x), y)
        first = first or loss
        model.backward(g)
        opt.step()
        last = loss
    assert last < 0.1 * first
