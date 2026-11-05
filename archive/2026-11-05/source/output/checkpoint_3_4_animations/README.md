# The life of the two packets: formal checkpoint Section 3.4

Three independent looping GIFs extend the two phase-offset packets from the formal checkpoint:

- `early_t0_to_8.gif`: 65 actual solution times, spacing 0.125.
- `middle_t8_to_64.gif`: 65 actual solution times, spacing 0.875.
- `late_t64_to_256.gif`: 65 actual solution times, spacing 3.

Each GIF plays its frames for 0.12 seconds and holds its final frame for 1.6 seconds. The stages therefore have different simulation-time playback speeds. Shared endpoints deliberately overlap. There is no interpolation between stored solution times.

## Reading the panels

The upper left is the y=0 slice in x,z. Its color scale is logarithmic and fixed across ALL stages: |u| divided by the initial peak, between 10^-5 and 1. This makes spreading ripples visible while preserving absolute decay relative to the start.

The upper right shows the central envelope as a height surface. Only this panel rescales height by the current sampled slice peak to make its shape readable. Its central viewing width is printed separately; it is narrower than the broad slice view at later times. The surface height is envelope magnitude, not water elevation.

The lower left gives the entire widening centerline, with a fixed logarithmic amplitude scale. The lower right keeps x between -12 and 12 so the carrier oscillations remain readable. Its vertical range is fixed within each stage (early ±1.1, middle ±0.65, late ±0.18), with numerical amplitude labels. The same illustrative carrier Re[u exp(3ix)] used in Section 3.4 appears inside ±|u|. It is not a measured physical water surface or fluid-particle trajectory.

The broad viewing halfwidth is 4 ceil[(24+4t)/4], increasing from 24 at t=0 to 1048 at t=256. Zoom changes appear in the numeric axes and title. It is a display window, not the computational period. The close-up and height surface prevent the central activity from becoming unreadably small in the broad view.

## What is being evolved

This is the original **linear canonical** model from the formal checkpoint, not the newer nonlinear solitary-wave solver:

    u_t + u_xxx - 3 u_xyy - 3 u_xzz = 0
    Fourier propagator: exp[it(kx^3 - 3 kx(ky^2+kz^2))].

Initial data:

    f = [exp(-((x+5)^2+y^2+z^2)/18)
         + 0.7i exp(-((x-5)^2+y^2+z^2)/18)] / Z.

Z is the original discrete L1 normalization on the 192^3 grid with spacing 0.75. It is held fixed at every time. The initial peak reference is evaluated at x=-5,y=z=0 on the continuous initial formula; the original figure used a slightly coarser sampled peak. No physical calibration or new decay exponent is inferred.

## Computing long times

The transverse Gaussian Fourier integrals are evaluated analytically. With r²=y²+z², d=9+6itk, and C=(2pi)^(3/2)27/Z, the remaining integral is

    u(t,x,r) = C/(4pi²) integral exp[-4.5k²+itk³+ikx-r²/(2d)]
                                    (exp(5ik)+0.7i exp(-5ik))/d dk.

A longitudinal FFT quadrature uses period 4096 and 16384 nodes, with 32768 nodes for the fine carrier close-up. Radial evaluation is nonperiodic. This avoids expanding a prohibitively large 3D FFT box as the field spreads. The numerical integration is separate from the rendering grid; very fine oscillations can become visually dense when the whole field is shown.

Independent checks at t=0,4,8,32,64,128,256 double the longitudinal period at fixed spacing and halve its spacing at fixed period. The largest checked relative complex-field change on the displayed sample grids is about 1.4e-8. Adaptive quadrature at each sampled peak agrees to better than 3e-9 relative error. Boundary magnitude on the broad view is below 0.3% of its sampled peak at these checked times; weak tails still extend beyond the view. These are sampled checks, not certification of a global supremum or all off-grid features.

The new evaluation also reproduces the original 192^3 computation at t=0,4,8 on selected shared slice points; the largest relative complex-field difference is about 3.7e-10. The apparent overlap and ripples represent linear interference and dispersion, not nonlinear attraction or a soliton collision.

## Reproduction and files

Run `OPENBLAS_NUM_THREADS=4 python animate_checkpoint_3_4.py` from the project root. Requires NumPy, SciPy, Matplotlib, and Pillow. The source and numerical checks are in the companion ZIP; all original reports remain unchanged. Frame metadata and validation files live in `results/checkpoint_3_4_animations/`.
