# Evaluation data

- `r2-1.png`: the MUTCD R2-1 "SPEED LIMIT 50" sign, Federal Highway Administration, U.S. Department of
  Transportation. **Public domain**. Rendered from
  [File:MUTCD_R2-1.svg](https://commons.wikimedia.org/wiki/File:MUTCD_R2-1.svg) on Wikimedia Commons at 960 px:
  ```
  curl -L -o data/eval/r2-1.png \
    "https://thumb.wikimedia.org/wikipedia/commons/thumb/8/8b/MUTCD_R2-1.svg/960px-MUTCD_R2-1.svg.png"
  ```
  Used by `src/scripts/eval_signs.py` as the template for synthetic speed-limit signs.
- `signs_results.json`: the latest output of `src/scripts/eval_signs.py`.

## Results (2026-10-22, 640 x 640 frames, `signs_results.json`)

- Stop approach (`20260927T184255Z-fddd`, seq 4000-4120): STOP in 26 of 83 frames.
- "False alarms": 4 in 8,001 frames, but all 4 are **real stop signs** at University Ave in
  `20261022T164127Z-084b` (seq 2646, 2661, 2676, 4512; checked by eye). The sweep assumed no stop
  signs outside the September approach, so the true false-alarm count is 0.
- Synthetic speed limits, 1,964 signs: read correctly 0% at 30-40 px wide, 33% at 40-50, 57% at
  50-70, 93% at 70-85, 99% at 85-100, 99% at 100-140; 3 misreads in total (0.15%).
