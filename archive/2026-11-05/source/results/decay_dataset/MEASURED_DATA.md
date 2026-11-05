# Measured-data candidate: CDIP buoy displacements

Status: source identified; **no measured record has been imported into the solver**.

CDIP at Scripps/UCSD provides displacement time series and quality flags:
https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/dw_timeseries.html

Deployment archive:
https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/catalog.html

The documentation's reproducible example is station 067, deployment 17,
21 May 2008 at 09:00 UTC, for 30 minutes. Variables include
`xyzZDisplacement`, `xyzFlagPrimary`, `xyzStartTime`, `xyzSampleRate` and
`xyzFilterDelay`. Record the exact deployment, retrieval date, checksum,
units, time correction, and quality filtering when retrieving a segment.

## Why this is not yet a 3D initial condition

The three displacement components describe a buoy's movement over time, not
samples of a field on a three-dimensional spatial grid. They do not determine
the initial field needed for the current equation. A Hilbert-transform envelope
of vertical displacement is still a one-dimensional time series.

Two defensible future uses are:

1. Use measured spectra or a processed envelope to construct explicitly
   **measurement-informed synthetic** fields, specifying the carrier removal,
   time-to-coordinate scaling, transverse profile, and random-phase assumptions.
2. Obtain spatially resolved envelope observations appropriate to the physical
   equation, with coordinate and unit conversion defined before importing them.

These categories must remain separate from direct measurements and from the
fully artificial batch. No optical-model validation can be claimed from simply
propagating a buoy-derived synthetic input.
