"""Small, unqualified learning models for the spatial water-wave Dysthe pilot.

Physics is supplied by :mod:`dysthe_core.torch_spatial`.  Both models enforce
the initial condition of the explicitly projected inlet, not an unresolved
input field.  Learned FNO modes and output Fourier bandwidth are separate.
All trainable parameters are real, so ``.float()``/``.double()`` also control
the precision of complex Fourier weights created during a forward pass.
"""

from collections.abc import Callable
import math

import torch
from torch import Tensor, nn

from dysthe_core.torch_spatial import linear_flow, project


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("activation must be 'gelu' or 'tanh'")


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _bandwidth(max_mode: int, size: int) -> None:
    # The Nyquist mode is excluded to keep a unique two-sided Fourier basis.
    if max_mode < 0 or 2 * max_mode >= size:
        raise ValueError("max_mode must satisfy 0 <= 2*max_mode < profile size")


def _batch_value(value: Tensor | float, batch: int, like: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if value.ndim == 0:
        return value.expand(batch)
    if value.shape == (batch,):
        return value
    raise ValueError(f"{name} must be scalar or have shape ({batch},)")


class FourierConv1d(nn.Module):
    """Low-mode convolution for real latent channels, including mode zero.

    The Hermitian completion belongs to each real *latent* channel.  The two
    final envelope channels remain independent real and imaginary fields.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        if min(in_channels, out_channels, modes) < 1:
            raise ValueError("channels and modes must be positive")
        self.modes = int(modes)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes, 2)
            / math.sqrt(in_channels * out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(x, dim=-1)
        count = min(self.modes, spectrum.shape[-1])
        weight = torch.view_as_complex(self.weight.contiguous())
        low = torch.einsum("bik,iok->bok", spectrum[..., :count], weight[..., :count])
        spectrum_out = torch.cat(
            (low, low.new_zeros(*low.shape[:-1], spectrum.shape[-1] - count)), dim=-1
        )
        return torch.fft.irfft(spectrum_out, n=x.shape[-1], dim=-1)


class ContinuousFNO(nn.Module):
    """Distance-conditioned map ``S_s P_K u0 + s P_K C_theta(u0,eps,s)``.

    ``u0`` has shape ``[batch, profile]`` and is complex; epsilon and distance
    are scalars or batch vectors.  The FFT operates only along the periodic
    profile.  Distance is a continuous input, enabling forward-mode AD.
    Inputs use the shared grid origin and physical (decoded) envelope scale.
    The default parameter dtype is float64; call ``.float()`` for complex64
    data.  Input and parameter precision must agree.
    """

    def __init__(self, length: float, width: int = 16, modes: int = 8,
                 depth: int = 2, max_mode: int = 24,
                 activation: str = "gelu", distance_scale: float = 1.0):
        super().__init__()
        if min(width, modes, depth) < 1 or max_mode < 0:
            raise ValueError("width, modes, depth must be positive; max_mode nonnegative")
        self.length = _positive(length, "length")
        self.distance_scale = _positive(distance_scale, "distance_scale")
        self.max_mode = int(max_mode)
        self.config = dict(length=self.length, width=width, modes=modes, depth=depth,
                           max_mode=self.max_mode, activation=activation,
                           distance_scale=self.distance_scale)
        self.lift = nn.Conv1d(4, width, 1)
        self.spectral = nn.ModuleList(FourierConv1d(width, width, modes) for _ in range(depth))
        self.local = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(depth))
        self.activation = _activation(activation)
        self.readout = nn.Conv1d(width, 2, 1)
        # Begin near the known linear evolution without disconnecting hidden
        # parameters from the first optimization step.
        with torch.no_grad():
            self.readout.weight.mul_(0.05)
            self.readout.bias.zero_()
        self.double()

    def forward(self, u0: Tensor, epsilon: Tensor | float, s: Tensor | float) -> Tensor:
        if not u0.is_complex() or u0.ndim != 2:
            raise ValueError("u0 must be a complex tensor of shape [batch, profile]")
        if u0.real.dtype != self.lift.weight.dtype or u0.device != self.lift.weight.device:
            raise ValueError("u0 and model parameters must have matching precision and device")
        _bandwidth(self.max_mode, u0.shape[-1])
        inlet = project(u0, self.max_mode)
        eps = _batch_value(epsilon, inlet.shape[0], inlet.real, "epsilon")
        distance = _batch_value(s, inlet.shape[0], inlet.real, "s")
        x = torch.stack((inlet.real, inlet.imag,
                         eps[:, None].expand_as(inlet.real),
                         (distance / self.distance_scale)[:, None].expand_as(inlet.real)), dim=1)
        x = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            x = self.activation(spectral(x) + local(x))
        channels = self.readout(x)
        correction = project(torch.complex(channels[:, 0], channels[:, 1]), self.max_mode)
        return linear_flow(inlet, distance, self.length) + distance[:, None] * correction


class SpectralPINN(nn.Module):
    """Per-inlet PINN: a smooth distance MLP predicts Fourier coefficients.

    ``forward(s)`` always returns ``[batch, profile]`` (batch one for scalar
    distance).  This fits one inlet and is not a learned operator over inlets.
    Coefficients use the DFT basis at the inlet array's shared grid origin.
    Real/imaginary inlet buffers permit safe precision conversion with
    ``.float()`` and ``.double()`` and are included in the state dictionary.

    ``coefficient_scaling='inverse_1_plus_k2'`` multiplies the learned
    correction coefficients by ``1 / (1 + (2*pi*m/length)**2)``. This is an
    invertible change of parameterization on the retained modes, not a filter
    on the inlet or a change to the PDE. The legacy default is ``'none'``.
    """

    def __init__(self, u0: Tensor, length: float, max_mode: int = 24,
                 width: int = 32, depth: int = 2, activation: str = "tanh",
                 distance_scale: float = 1.0, coefficient_scaling: str = "none"):
        super().__init__()
        if u0.ndim == 2 and u0.shape[0] == 1:
            u0 = u0[0]
        if not u0.is_complex() or u0.ndim != 1:
            raise ValueError("u0 must be one complex inlet, shape [profile] or [1, profile]")
        if min(width, depth) < 1:
            raise ValueError("width and depth must be positive")
        if coefficient_scaling not in ("none", "inverse_1_plus_k2"):
            raise ValueError("coefficient_scaling must be 'none' or 'inverse_1_plus_k2'")
        _bandwidth(max_mode, u0.shape[-1])
        self.length = _positive(length, "length")
        self.distance_scale = _positive(distance_scale, "distance_scale")
        self.max_mode = int(max_mode)
        self.coefficient_scaling = coefficient_scaling
        self.config = dict(length=self.length, max_mode=self.max_mode, width=width,
                           depth=depth, activation=activation,
                           distance_scale=self.distance_scale,
                           coefficient_scaling=self.coefficient_scaling)
        inlet = project(u0.detach(), self.max_mode)
        self.register_buffer("inlet_real", inlet.real.clone())
        self.register_buffer("inlet_imag", inlet.imag.clone())
        self.register_buffer("mode_indices", torch.arange(-self.max_mode, self.max_mode + 1,
                                                          device=u0.device) % u0.shape[-1])
        # Integer modes recompute the scale in the current real dtype, including
        # after .float().double(); no rounded float32 scale is silently reused.
        self.register_buffer("correction_modes", torch.arange(-self.max_mode, self.max_mode + 1,
                                                               device=u0.device), persistent=False)
        # Legacy unscaled state dictionaries remain strictly loadable. Scaled
        # checkpoints carry a marker so a strict load into an unscaled model
        # cannot silently change the trial function (or vice versa).
        self.register_buffer("_coefficient_scaling_code",
                             torch.tensor(1, device=u0.device) if coefficient_scaling != "none" else None)
        layers: list[nn.Module] = []
        for index in range(depth):
            layers.extend((nn.Linear(1 if index == 0 else width, width), _activation(activation)))
        layers.append(nn.Linear(width, 2 * (2 * self.max_mode + 1)))
        self.network = nn.Sequential(*layers).to(dtype=u0.real.dtype, device=u0.device)
        with torch.no_grad():
            self.network[-1].weight.mul_(0.01)
            self.network[-1].bias.zero_()

    @property
    def u0(self) -> Tensor:
        return torch.complex(self.inlet_real, self.inlet_imag)

    def forward(self, s: Tensor | float) -> Tensor:
        distance = torch.as_tensor(s, dtype=self.inlet_real.dtype, device=self.inlet_real.device)
        if distance.ndim == 0:
            distance = distance[None]
        if distance.ndim != 1:
            raise ValueError("s must be scalar or a one-dimensional batch")
        pairs = self.network((distance / self.distance_scale)[:, None])
        coefficients = torch.view_as_complex(pairs.reshape(distance.shape[0], -1, 2).contiguous())
        if self.coefficient_scaling == "inverse_1_plus_k2":
            k = self.correction_modes.to(dtype=self.inlet_real.dtype) * (2 * math.pi / self.length)
            coefficients = coefficients / (1 + k.square())
        spectrum = coefficients.new_zeros(distance.shape[0], self.inlet_real.shape[0])
        spectrum = spectrum.index_copy(-1, self.mode_indices, coefficients)
        correction = torch.fft.ifft(spectrum, dim=-1, norm="forward")
        inlet = self.u0[None].expand(distance.shape[0], -1)
        return linear_flow(inlet, distance, self.length) + distance[:, None] * correction


def value_and_distance_derivative(predict: Callable[[Tensor], Tensor], s: Tensor) -> tuple[Tensor, Tensor]:
    """Return field and d(field)/ds for independent batch entries.

    ``predict`` must not mix distances across its batch.  Unlike a gradient
    of the sum of output entries, this JVP retains every complex field value
    and remains differentiable with respect to the model parameters.
    """
    return torch.func.jvp(predict, (s,), (torch.ones_like(s),))


def rollout(model: ContinuousFNO, u0: Tensor, epsilon: Tensor | float,
            step: float, steps: int) -> Tensor:
    """Differentiable autonomous rollout; includes the projected initial field.

    No reference reset or detach is performed.  The returned shape is
    ``[batch, steps + 1, profile]``.  Campaigns choose their own rollout loss.
    """
    _positive(step, "step")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    field = project(u0, model.max_mode)
    fields = [field]
    for _ in range(steps):
        field = model(field, epsilon, step)
        fields.append(field)
    return torch.stack(fields, dim=1)
