"""Differentiable operators for ``water-wave-dysthe-spatial-v1`` only.

Profiles occupy the last tensor axis on the periodic grid
``tau[j] = -length/2 + j*length/n``; leading axes are batch axes. Import this
module explicitly: the dependency-free package entry point does not load torch.
No operation detaches a field or differentiable scalar from its autograd graph.

The full RHS is a pseudospectral evaluation on the supplied grid, not an
automatic continuum residual certificate. Oversample bandlimited fields before
evaluating nonlinear products when all generated output modes are required.
"""

import math
from numbers import Integral, Real

import torch

__all__ = ["derivative", "abs_derivative", "rhs", "linear_flow", "project", "resample"]


def _field(u, *, complex_only=False):
    allowed = (torch.complex64, torch.complex128)
    if not complex_only:
        allowed += (torch.float32, torch.float64)
    if not isinstance(u, torch.Tensor) or u.dtype not in allowed:
        raise TypeError("Expected a floating-point or complex torch profile"
                        if not complex_only else "Expected a complex torch profile")
    if u.ndim < 1 or u.shape[-1] < 1:
        raise ValueError("A nonempty last profile axis is required")
    return u


def _length(length):
    if isinstance(length, bool) or not isinstance(length, Real):
        raise ValueError("length must be a finite positive real number")
    if not math.isfinite(length) or length <= 0:
        raise ValueError("length must be a finite positive real number")
    return float(length)


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _modes(u):
    n = u.shape[-1]
    indices = torch.arange(n, device=u.device)
    return torch.where(indices < (n + 1) // 2, indices, indices - n)


def _k(u, length):
    return _modes(u).to(u.real.dtype) * (2 * math.pi / _length(length))


def _parameter(value, u, name, *, nonnegative=False):
    # Avoid tensor-to-Python value checks: these break batched transforms and
    # would synchronize devices on every residual evaluation. Tensor-valued
    # parameters must be finite (and epsilon nonnegative) at the caller boundary.
    if isinstance(value, torch.Tensor):
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{name} must be real")
        value = value.to(dtype=u.real.dtype, device=u.device)
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be real")
        if not math.isfinite(value) or (nonnegative and value < 0):
            raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
        value = torch.as_tensor(value, dtype=u.real.dtype, device=u.device)
    try:
        shape = torch.broadcast_shapes(value.shape, u.shape[:-1])
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast over the existing batch axes") from exc
    if shape != u.shape[:-1]:
        raise ValueError(f"{name} must broadcast over the existing batch axes")
    return value[..., None]


def derivative(u, length, order=1):
    """Periodic profile derivative, preserving float/complex precision.

    For complex fields the even-grid Nyquist bin uses the negative FFT mode.
    Real input returns the real derivative of its trigonometric interpolant.
    """
    _field(u)
    order = _integer(order, "order", 0)
    multiplier = (1j * _k(u, length)) ** order
    result = torch.fft.ifft(multiplier * torch.fft.fft(u, dim=-1), dim=-1)
    return result if u.is_complex() else result.real


def abs_derivative(u, length):
    """Apply ``|D_tau| = H d_tau`` with symbol ``|k|`` and zero constant mode."""
    _field(u)
    result = torch.fft.ifft(_k(u, length).abs() * torch.fft.fft(u, dim=-1), dim=-1)
    return result if u.is_complex() else result.real


def project(u, max_mode):
    """Retain integer Fourier modes ``|m| <= max_mode`` (inclusive)."""
    _field(u)
    max_mode = _integer(max_mode, "max_mode", 0)
    if max_mode > u.shape[-1] // 2:
        raise ValueError("max_mode cannot exceed the grid Nyquist index")
    result = torch.fft.ifft(torch.fft.fft(u, dim=-1)
                            * (_modes(u).abs() <= max_mode), dim=-1)
    return result if u.is_complex() else result.real


def linear_flow(u, s, length):
    """Exact flow of ``u_xi = -i u_tautau`` over distance ``s``.

    ``s`` may be a real scalar or tensor broadcastable to the existing batch
    shape. Negative distance is permitted. Tensor values must be finite.
    """
    _field(u, complex_only=True)
    distance = _parameter(s, u, "s")
    phase = torch.exp(1j * distance * _k(u, length).square())
    return torch.fft.ifft(phase * torch.fft.fft(u, dim=-1), dim=-1)


def rhs(u, epsilon, length, project=False):
    """Approved free-surface Dysthe RHS, with coefficients ``(1,1,8,2,2)``.

    ``epsilon`` may be a scalar or real tensor broadcastable to the existing
    batch shape. Tensor parameters must be finite and nonnegative; validation
    of their values belongs at the caller boundary to preserve torch transforms.

    The default applies no output projection, so PINN defects retain generated
    modes. ``project=True`` projects the *whole output* to ``|m| < n/4``.
    On retained-band input this agrees with ``SpatialDysthe.rhs_hat``; that
    reference otherwise leaves unretained linear input modes untouched.
    Neither option projects the input or automatically pads nonlinear products.
    """
    _field(u, complex_only=True)
    if not isinstance(project, bool):
        raise TypeError("project must be a boolean")
    eps = _parameter(epsilon, u, "epsilon", nonnegative=True)
    k = _k(u, length)
    spectrum = torch.fft.fft(u, dim=-1)
    du = torch.fft.ifft(1j * k * spectrum, dim=-1)
    linear = torch.fft.ifft(1j * k.square() * spectrum, dim=-1)
    rho = u.real.square() + u.imag.square()
    mean_flow = torch.fft.ifft(k.abs() * torch.fft.fft(rho, dim=-1), dim=-1).real
    result = linear - 1j * rho * u - eps * (
        8 * rho * du + 2 * u.square() * du.conj() + 2j * u * mean_flow)
    if project:
        result = torch.fft.ifft(torch.fft.fft(result, dim=-1)
                                * (_modes(u).abs() < u.shape[-1] / 4), dim=-1)
    return result


def resample(u, n):
    """Fourier interpolate/project to ``n`` samples at the same periodic origin.

    Normalized Fourier amplitudes are preserved. Downsampling truncates high
    modes rather than aliasing them. At an even-grid Nyquist mode, interpolation
    is ambiguous: use the standard symmetric split on upsampling and merge on
    downsampling, matching ``scipy.signal.resample`` for periodic profiles.
    """
    _field(u)
    n = _integer(n, "n", 1)
    old_n = u.shape[-1]
    if n == old_n:
        return u
    spectrum = torch.fft.fft(u, dim=-1)
    result_hat = spectrum.new_zeros((*u.shape[:-1], n))
    common = min(n, old_n)
    positive = common // 2 + 1
    negative = (common - 1) // 2
    result_hat[..., :positive] = spectrum[..., :positive]
    if negative:
        result_hat[..., -negative:] = spectrum[..., -negative:]
    if common % 2 == 0:
        if n < old_n:
            result_hat[..., common // 2] = (spectrum[..., common // 2]
                                             + spectrum[..., -common // 2])
        else:
            result_hat[..., common // 2] = spectrum[..., common // 2] * .5
            result_hat[..., -common // 2] = spectrum[..., common // 2] * .5
    result = torch.fft.ifft(result_hat, dim=-1) * (n / old_n)
    return result if u.is_complex() else result.real
