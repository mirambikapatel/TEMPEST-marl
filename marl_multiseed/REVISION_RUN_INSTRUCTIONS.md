# TEMPEST revision experiments — how to run

New files (nothing existing was modified):

| File | What it is |
|---|---|
| `baselines.py` | QDOM-adapted (ref 36) and DRLE-adapted (ref 37) baselines, background user behaviours, refuted/spontaneous tagging, signed polarization |
| `revision_runner.py` | Runs every revision experiment in priority order, in parallel, resumable |
| `summarise_v2.py` | Turns results into tables (mean ± SD, paired t-tests vs TEMPEST) |

Results go to `results_v2/`. **`multiseed_results/` is never touched.**

## 1. Smoke test first (~10–20 min)

Open Terminal, then:

```bash
cd marl_multiseed
python revision_runner.py --smoke --workers 4
python summarise_v2.py --outdir results_v2_smoke
```

Every line should end in `ok`. If any says `ERROR`, open the log it names
(`results_v2_smoke/<exp>/<variant>/logs/seed_000.log`) and check the last ~20 lines for the error.

## 2. Full run (overnight)

```bash
cd marl_multiseed
caffeinate -i python revision_runner.py --workers 5 --threads 2 2>&1 | tee results_v2_run.log
```

* `caffeinate -i` stops the Mac from sleeping while it runs. Keep it plugged in.
* **Resumable:** if anything stops (sleep, crash, you close Terminal), run the same command again.
  Finished jobs are skipped.
* See the plan without running: `python revision_runner.py --list`
* Run just some experiments: `python revision_runner.py --only main ablation`
* Optional extras (T retraining sweep, ε-decay check): add `--include-optional`

Priority order (most important first): main → ablation → networks → sensitivity → k_sweep →
baselines_extra → sir_grid → (optional) T_sweep, eps_decay.

## 3. Check progress any time

```bash
python revision_runner.py --list          # how many jobs are left
python summarise_v2.py                    # tables from whatever has finished so far
```

## 4. If the Mac gets too hot / slow

Stop with Ctrl+C, then restart with fewer workers: `--workers 3`. Nothing is lost.
