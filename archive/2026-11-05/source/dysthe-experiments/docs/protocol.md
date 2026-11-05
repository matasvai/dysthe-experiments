# Promotion gates

1. **Reference:** choose parameter bounds and horizon; independently check propagation-step,
   grid and domain sensitivity, projection error, and resolved spectrum.
2. **Dataset:** freeze complete initial-condition groups across train/validation/
   test; include family holdouts. Hash arrays and store exact core commits.
3. **Execution smoke:** verify imports, tiny data generation, training, checkpoint
   resume and evaluation on the intended environment. A pass is not accuracy.
4. **Model selection:** tune using training and validation only. Use validation
   rollouts for early stopping, record the stopping rule and selected epoch.
5. **Final evaluation:** freeze weights and settings before touching the test set.
   Compare complex-field and phase errors, weighted profile L1/L2/L4/Linf over tau,
   growth/focusing distance, invariant drift, spectral errors and runtime. Report
   per-case distributions and multiple seeds, including failures. Set numerical
   acceptance thresholds before inspecting final test results.
6. **Paper:** promote only traceable results with data/config/checkpoint hashes,
   code commits, hardware, software versions and solver accuracy settings.

Measure numerical solver cost at matched error as well as model inference cost.
Report data-generation and training cost separately. For active GP, use scalar
error, calibration and simulation budget; its target differs from field models.

Large artifacts live in durable cluster/object storage. Git holds compact
manifests, metrics and vector figures. Checksums identify artifacts; a checksum
alone is not a backup or a usable access location.
