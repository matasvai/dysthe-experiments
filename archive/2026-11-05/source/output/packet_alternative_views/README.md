# Alternative views of the formal checkpoint's two packets

These views use the same linear canonical equation, Gaussian packets, fixed normalization Z, and analytically reduced numerical evolution as the previous animations. They are not the nonlinear Dysthe collision experiment.

## Views

- `moving_closeup.gif`: 73 computed times, from 0 through 256. The strongest sampled centerline peak is followed inside a fixed ±20 window. The left image divides by the current local peak to show shape; the right plot divides by the initial peak and keeps a fixed vertical scale to show decay. This tracks a maximum, not a material particle or a tagged crest. Time spacing changes after t=8 and t=64; the displayed times are authoritative. The searched peak-position interval is [-24,100].
- `space_time.png`: 369 computed time rows, on an actual linear time axis. Both panels use a fixed logarithmic scale |u|/initial peak. The right panel enlarges the central region. These are centerline histories, not whole-volume maxima; weak outer tails extend beyond the plotted x range.
- `phase_surface.png`: at t=0,8,64, height represents magnitude/current peak and cyclic color represents complex phase. Each panel's spatial range is labeled by its axes. Surface points below 1% of that panel's peak are omitted because phase is uninformative near zero amplitude.
- `interference_control.png`: top row is the ORIGINAL pair with a 90-degree initial phase offset. Bottom row is an ADDITIONAL control with that offset changed to zero, holding the component amplitudes and Z fixed. The control is not independently renormalized to L1=1. The plotted dimensionless contrast is 2 Re(u1 conjugate(u2))/(|u1|²+|u2|²). Positive red reinforces and negative blue cancels. Regions where component-sum intensity is below 10^-5 of its peak are masked. The original pair's contrast is zero to roundoff; the control can approach either ±1.
- The inline rotatable 3D view shows actual magnitude LEVEL SURFACES in (x,y,z), rather than a height plot. It uses radial symmetry in y,z, revolving computed contours from the (x,r) plane. Times are 0,8,64,256 and thresholds are 10%,30%,60% of each current sampled peak. The camera fits each selected surface, so its screen size does not measure decay. The peak ratio and coordinate-axis sizes remain visible. Geometry is resampled for interactive display and is not a new supremum computation.

## Correction to the earlier interpretation

The original formal report and previous discussion called the quadrature pair an interference example and suggested that its intensity cross-term could become nonzero. That claim is incorrect for this particular linear canonical propagator.

The equation u_t+u_xxx-3u_xyy-3u_xzz=0 has real coefficients and preserves real-valued data. Equivalently, its Fourier symbol w is odd, so its evolution kernel is real. If a and b are the real evolved versions of the two real Gaussian profiles, the original solution has the form u=a+0.7 i b (including the same normalization). Thus

    |u|² = a² + 0.49 b²,
    2 Re[a conjugate(0.7 i b)] = 0

at EVERY time, not only initially. Changes in the original magnitude picture reflect spreading and overlap of these component intensities; they are not mutual constructive/destructive intensity interference. The new decomposition checks give contrast below 1.6e-14 on the unmasked plotted regions. The phase-colored combined field can still have a spatially varying argument.

For the zero-phase control u=a+0.7 b, the cross-term is 1.4 ab, which can be positive or negative after the real components develop oscillatory tails. The bottom row of the control figure visualizes that distinct initial condition explicitly.

This correction changes the interpretation of the original chosen pair, not the previously computed total-field evolution. The older compiled checkpoint has not been silently replaced; this note records the correction for the next report revision.

## Reproduction

Run `OPENBLAS_NUM_THREADS=4 python make_packet_alternative_views.py`, followed by `OPENBLAS_NUM_THREADS=4 python make_interference_control.py` from the project root. These use `animate_checkpoint_3_4.py` for the integral evaluator and normalization. PNGs, plot PDFs, the moving GIF, level-contour data, and decomposition diagnostics are retained here. The plots and core data are included in the companion ZIP. The inline viewer is supplied separately in the conversation.
