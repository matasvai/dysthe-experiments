# Custom initial envelopes

The custom formula describes the **complex initial envelope** `u(0, tau)` of the
existing spatial free-surface water-wave Dysthe model. Enter only the formula's
right-hand side, then declare the real scalar parameters it uses. It is not a
formula for reconstructed surface elevation, a change of equation, or a claim
that this particular packet is numerically or physically validated.

Preview first to inspect amplitude and phase, the periodic boundary, and the
resolution/regime screens. A successful preview or run is exploratory and does
not establish convergence. Boundary mismatch, insufficient decay, high spectral
content and excessive steepness or bandwidth require revising the input or a
separate convergence study. A grid can also alias unresolved oscillations; the
screens are diagnostics rather than a proof of resolution.

Open **Custom formula** in the workbench, enter a formula below and use scalar
declarations such as `A=1, w=2, k=0.3, c=0` in the Parameters field. Click
**Preview initial envelope**, inspect amplitude and masked phase, then **Run
locally**. Changing the formula, any parameter, grid or propagation interval
invalidates that preview immediately. The JSON declarations below are the
equivalent API notation, rather than text to paste into the Parameters field.

## Working examples

The default reproduces the existing Gaussian initializer, including both complex
channels:

```latex
A \exp(-\frac{\tau^2}{2 w^2}) \exp(i(k\tau+c\tau^2))
```

```json
{"A": 1, "w": 2, "k": 0.3, "c": 0}
```

A displaced sech packet with chirped phase:

```latex
A \sech(\frac{\tau-b}{w}) \exp(i(k\tau+c\tau^2))
```

```json
{"A": 1.1, "w": 1.8, "b": 0.4, "k": -0.3, "c": 0.02}
```

A Gaussian with mild amplitude modulation and a phase offset:

```latex
A \exp(-\frac{\tau^2}{2 w^2}) (1+b \cos(q\tau)) \exp(i(k\tau+p))
```

```json
{"A": 0.8, "w": 2, "b": 0.1, "q": 0.5, "k": 0.2, "p": 0.4}
```

These expressions are accepted syntax examples, not independently qualified
reference cases. The unchanged saved examples retain their own qualification.

## Exact supported syntax

| Input | Meaning |
| --- | --- |
| `tau`, `\tau` | Retarded-time coordinate on the selected periodic grid |
| `i` | Imaginary unit |
| `pi`, `\pi` | Numerical value of pi |
| `A`, `w`, `k`, etc. | Declared single ASCII-letter real scalar parameter; case sensitive |
| `0.3`, `.3`, `2`, `1e-3` | Decimal or scientific-notation number |
| `+`, `-`, `*`, `/` | Addition, subtraction, multiplication, division |
| `A tau`, `A(tau+b)`, `k\tau` | Implicit multiplication, including a parameter followed by a group |
| `\cdot`, `\times` | Multiplication |
| `tau^2`, `tau^{-2}`, `tau^(-2)` | Literal integer powers from -12 through 12 |
| `( ... )`, `{ ... }` | Grouping |
| `\left( ... \right)` | Optional scalable round delimiters |
| `\frac{a}{b}` | Division with two required braced arguments |
| `exp(...)`, `\exp(...)` | Complex exponential |
| `sin(...)`, `\sin(...)` | Complex sine |
| `cos(...)`, `\cos(...)` | Complex cosine |
| `sech(...)`, `\sech(...)` | Reciprocal complex hyperbolic cosine |

Functions require a parenthesized or braced argument. A parameter followed by a
group is multiplication, not a custom function. Write `exp(...)`; the letter `e`
is not a built-in Euler constant. If declared, `e` is just another parameter.

Powers bind more tightly than unary signs: `-tau^2` means `-(tau^2)`.
Multiplication, implicit multiplication and division have equal precedence and
associate left to right: `1/2 tau` means `(1/2)*tau`; use `1/(2 tau)` when intended.
Addition and subtraction also associate left to right. Powers cannot be chained;
write explicit grouping. Noninteger and variable exponents are unsupported.

Enter no assignment (`u(0,tau)=`), dollar delimiters, LaTeX environment, indices,
derivative, integral, summation, arbitrary function or command. This is a small
formula language, not a general LaTeX or symbolic mathematics engine. The exact
allowlist above is the supported interface; Python and JavaScript are never
evaluated. There is no filesystem access, dynamic attribute lookup or symbolic
algebra in evaluation.

## Bounds, errors and provenance

There may be at most eight declared parameters. Names must be single ASCII
letters except reserved `i`; values must be actual finite real numbers rather
than booleans, strings, complex numbers, arrays or lists. Their magnitudes and
numeric literals are limited to `1e6`. Numeric literals have at most 32 characters.
Expressions have at most 1,024 characters, 512 tokens, 256 expression nodes and
32 levels of grouping, unary nesting or expression-tree depth. All integer
powers have magnitude at most 12.

Evaluation accepts one real finite axis of 1 through 16,384 samples with
`|tau| <= 1e6`. Each intermediate value must remain finite with magnitude at most
`1e12`. Division by zero, overflow and invalid operations produce an understandable
formula error. Underflow to zero in a decaying tail is allowed. Scalar formulas
broadcast to the entire axis, and the result is always an independent complex128
array. A zero profile is representable by the parser but cannot be propagated
when its zero action makes the simulator's normalized diagnostics undefined.

Stored formula provenance includes the original text, declared scalar values,
an explicit normalized expression, parser version and parser bounds. The full
complex initial field and propagated real/imaginary data must be retained with
the simulation output; a magnitude plot alone does not specify the input.

The downloaded `.npz` contains `u` (complex128, saved xi by tau), `initial`
(projected complex128 input), `raw_initial` (formula sampled before projection),
`tau`, `xi`, and scalar Unicode `provenance_json`. Read it without pickle:

```python
import json
import numpy as np

with np.load('dysthe-complex-fields.npz', allow_pickle=False) as data:
    u = data['u']
    provenance = json.loads(str(data['provenance_json']))
```

Provenance records the formula, symbols, parser/model/source hashes, exact
settings, actual propagation step, initial-field checks and warnings, and raw
and projected initial-field hashes. The accompanying server `report.json` and
viewer response also retain it. Formula source hashes bind a preview fingerprint
to the input, settings and implementation; the server resamples and rechecks on
Run rather than trusting that fingerprint as validation.

## Preview and local-run limits

Custom UI defaults are N=1024, periodic tau length=160, epsilon=0.025,
requested step=0.00125, xi end=0.1 and up to 21 saved snapshots. The UI offers
N=1024 or 4096 and xi end through 4; the API accepts multiples of four from 128
through 8192, length 40 through 640, step 0.0003125 through 0.005, xi end in (0,4]
and 2 through 101 requested snapshots. All custom runs additionally satisfy
`ceil(xi_end/dxi)*N <= 8,000,000`; shorten the horizon when that limit is reached.
The actual step is xi_end divided by the integer step count. No refinement
campaign is automatically launched for a custom formula.

Preview samples the formula at N and independently at 2N, compares the latter
with Fourier interpolation from N, and inspects the spectrum against the actual
strict `abs(mode)<N/4` projector. Run is blocked if projection or 2N discarded-band
relative norm exceeds 1e-8, or the cross-grid discrepancy exceeds 1e-6. Appreciable
retained-edge action produces a warning. These finitely many samples can still
miss unresolved behavior and do not replace a convergence study.

Sampled peak amplitude must lie between 1e-12 and 4. Initial epsilon times peak,
spectral standard deviation and absolute spectral mean may not exceed 0.1.
These are chosen local demo screens, not physical validity theorems. Run is
also blocked for nonfinite or undefined evaluations. Boundary action above 1e-8,
relative boundary amplitude above 1e-6 or endpoint mismatch above 1e-6 produces
periodic-window warnings; a constant or broad periodic field can still run.
Near-zero circular concentration makes packet center, width, exterior action,
translation/phase fit and displacement unavailable rather than inferred from
roundoff. Broad or split packets can make these diagnostics ambiguous even
above the near-zero cutoff.

`POST /api/preview` accepts `{"case_id":"custom","parameters":{"formula":
"...","symbols":{"A":1}},"settings":{...}}` and returns the preview, warnings,
`can_run` and a fingerprint. `POST /api/run` accepts the same request plus
`preview_fingerprint` from that exact successful preview. Errors return a clear
HTTP 400 JSON `error`; busy work returns 409. No output directory or solver job
is created by a preview.

The evaluator API is:

```python
from dysthe_experiments.envelope_formula import compile_formula

compiled = compile_formula(
    r'A \sech(\tau/w) \exp(i k \tau)',
    {'A': 0.8, 'w': 2, 'k': 0.2},
)
initial = compiled.evaluate(tau)  # 1D complex128, sampled on the chosen grid
provenance = compiled.provenance()  # JSON-safe independent dictionary
```

Syntax and declaration errors raise `FormulaError`, a `ValueError` subclass,
during compilation. Numerical and axis errors raise it during sampling.
