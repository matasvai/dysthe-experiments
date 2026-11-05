Custom LaTeX initial envelopes — installed and verified 2026-10-16

Open http://127.0.0.1:8765/ on Fedora and select Custom formula. Refresh an old tab.
Enter the formula and scalar declarations, Preview initial envelope, inspect warnings, then Run locally on CPU. Gaussian and sech buttons insert supported examples.

Exact syntax, tested examples, bounds, API and complex-export reading instructions: docs/custom-envelope.md.
Explanation of scaled downstream xi, moving-frame retarded time tau and the three different refinement checks: docs/small-simulator.md.

Evidence records 13 parser tests, 9 custom integration tests, 6 simulator regression tests and 16 installed-browser checks passed. Only tiny smoke integrations were used for this feature; no custom convergence campaign was run. Custom inputs remain exploratory.

Screenshots are from the installed server at desktop and mobile widths. tiny-custom-example.npz is the actual browser-exported four-step example, complex128 shape (5,1024), with raw/projected inputs and scalar Unicode JSON provenance. It is a UI/export smoke result, not a validated reference trajectory.

The existing loopback-only transient user service remains running. No firewall or boot-autostart setting was changed. Restart after logout/reboot as documented previously; this is not a public website.

Reviewed source changes are listed in evidence/install-manifest.json. The original equation/core and unrelated working files were preserved.
