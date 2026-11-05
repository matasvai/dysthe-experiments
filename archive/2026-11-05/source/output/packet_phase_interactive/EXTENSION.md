# Phase viewer extension through t=2056

The viewer now has 122 independently computed snapshots:
- t=0 to 16, step 1;
- 18 to 64, step 2;
- 72 to 256, step 8;
- 288 to 1024, step 32;
- 1056 to 2048, step 32;
- final snapshot at 2056.

Slider positions index these samples; they are not uniformly spaced in time. The time label gives the actual evolution coordinate. This remains the same linear canonical two-packet solution.

The integral evaluator uses longitudinal period 4096 through t=256, then increases it by powers of two with time. At t=2056 it uses period 65536 and 262144 Fourier nodes. Transverse Gaussian integration remains analytic. Domain-doubling and spacing-halving checks on identical displayed coordinates are stored in extended_checks.json. The largest measured spacing-change field discrepancy at the checked times is 0.0302%; at 2056 it is 0.00162%. These are numerical consistency checks, not rigorous error bounds.

The central view follows the prior extent 16+0.15t through 256 and then expands proportionally to (t/256)^(1/3), reaching approximately ±109. This keeps the core readable on the 65×65 display mesh; it is not a picture of all outgoing tails. The printed peak ratio refers to the sampled central view.

The numerical calculation uses complex double precision. For compact interactive display, magnitude and phase are quantized to 8 bits each. Maximum magnitude rounding is 1/510 of the current view peak; maximum phase rounding is pi/255 radians (about 0.71 degrees). Reflection symmetry stores only half the radial slice. The underlying calculation is not reduced to 8-bit precision. There is no temporal interpolation.

Reproduce the data with `OPENBLAS_NUM_THREADS=4 python output/packet_phase_interactive/extend_data.py` from the project root. Select “Initial peak” in the viewer to see actual amplitude decay rather than normalized shape.
