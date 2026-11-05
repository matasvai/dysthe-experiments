# Browser verification

The workbench was tested in Firefox 157.0.1 with a fresh temporary profile, using
only the local simulator. Desktop (1440 × 1000) and narrow (390 × 844) renders
were inspected. No personal Firefox profile was used.

- `browser-e2e.json`: full UI flow on the staging server, including all three
  cases, field/domain/heatmap modes, scrub/play/pause, a heatmap click, one fresh
  custom Gaussian run (`A=.8`, `width=2.2`, `wavenumber=.25`, `chirp=.01`), its
  unqualified-result label, and a browser-triggered NPZ download. The downloaded
  field array is complex128 with shape `(81, 4096)`.
- `installed-project/browser-e2e.json`: final verification of the installed
  project at `http://127.0.0.1:8766/`, after matching all three frontend hashes.
  This pass made no new numerical run. All 12 executed checks passed. Includes
  sampled rendered-text contrast ratios of 5.40–12.00:1, no horizontal overflow
  for any case at 390px, and no browser script errors.
- `installed-project/*.png`: final installed-project screenshots, including
  amplitude, phase, heatmap and narrow layouts. `desktop-live-custom-gaussian.png`
  shows the separate live-result flow from the full test.
- `focus-and-layout-check/`: intermediate focus probes. They found a headless
  environment limitation, not a numerical failure; the final pass records it
  explicitly below.

Headless Firefox reports `document.hasFocus() = false`, even after explicitly
activating the context and setting Mozilla's recommended `focusmanager.testmode`
preference. Actual Tab input moves `activeElement` correctly, but CSS focus
pseudo-classes are not active. Focus-outline rendering therefore remains
unverified in this environment. The shipped UI provides explicit 2px `:focus`
outlines; a normal interactive browser should receive a brief keyboard check.

To repeat the final UI checks without running the solver again:

```sh
python stage2/qa/browser_e2e.py --skip-live \
  --url http://127.0.0.1:8766/ \
  --output stage2/qa/recheck
```

Omitting `--skip-live` launches **one** bounded local custom Gaussian through the
actual UI. The script needs already-installed Firefox, Python `websockets` and
NumPy. It automatically retires its temporary Firefox process and leaves the
simulator server running. It opens no external webpage. Protocol references:
[Mozilla Remote Protocols](https://firefox-source-docs.mozilla.org/remote/) and
[WebDriver BiDi](https://w3c.github.io/webdriver-bidi/).
