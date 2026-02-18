# Automation Templates

This folder provides a runnable scaffold for the plan documented in:

- `AUTO_EXPERIMENT_MASTER_PLAN.md`

## Files

- `run_single_experiment.py`: one standardized trial run
- `run_sweep.py`: multi-trial sweep with timeout/resume/topK
- `run_pair_runner.py`: async simulator/planner process runner
- `classify_failure.py`: trial label classification
- `propose_next_params.py`: propose next parameter set
- `analyze_results.py`: summary report generation
- `server_submit_sweep.sh`: launch sweep on Ubuntu server over SSH
- `server_fetch_results.sh`: rsync sweep results back to local machine
- `sweep_space.yaml`: parameter sampling space
- `policy_rules.yaml`: scoring/classification/rule adjustments
- `sweep_utils.py`: shared helper utilities

## Quick Start

Install minimal dependencies:

```bash
python -m pip install pyyaml
```

Recommended simulation stack (matches `dial-mpc/versions_before.txt`):

```bash
python -m pip install --user \
  'brax==0.14.0' \
  'jax==0.6.2' \
  'jaxlib==0.6.2' \
  'mujoco==3.4.0' \
  'mujoco-mjx==3.4.0' \
  'jaxopt==0.8.5' \
  'jax-cosmo==0.1.0' \
  art scienceplots emoji tyro tqdm
```

Run a smoke sweep (dry run):

```bash
python automation/run_sweep.py \
  --config dial-mpc/dial_mpc/examples/sds_gallop_sim.yaml \
  --space automation/sweep_space.yaml \
  --out runs/sweep_smoke \
  --trials 3 \
  --topk 2 \
  --dry-run
```

Run a real sweep (uses default runner command):

```bash
python automation/run_sweep.py \
  --config dial-mpc/dial_mpc/examples/sds_gallop_sim.yaml \
  --space automation/sweep_space.yaml \
  --out runs/sweep_$(date +%F) \
  --trials 20 \
  --topk 5 \
  --timeout-sec 1800 \
  --resume
```

Default runner now launches:

- `automation/run_pair_runner.py`
- simulator: `dial-mpc/dial_mpc/deploy/dial_sim.py`
- planner: `dial-mpc/dial_mpc/deploy/dial_plan.py`

`run_single_experiment.py` injects safe batch defaults into config when missing:

- `headless=true`
- `max_time_sec=12.0`
- `stop_on_fall=true`
- `metrics_filename=metrics.json`

Analyze results:

```bash
python automation/analyze_results.py --results runs/sweep_YYYY-MM-DD/results.csv
```

Propose next params:

```bash
python automation/propose_next_params.py \
  --results runs/sweep_YYYY-MM-DD/results.csv \
  --space automation/sweep_space.yaml \
  --policy-rules automation/policy_rules.yaml \
  --out automation/proposed_params.yaml
```

Run from MacBook to Ubuntu server:

```bash
chmod +x automation/server_submit_sweep.sh automation/server_fetch_results.sh
./automation/server_submit_sweep.sh <server> <remote_repo_dir> <python_bin> 200
./automation/server_fetch_results.sh <server> <remote_repo_dir> sweep_YYYY-MM-DD ./runs
```

## Important Notes

- Current project still needs reward/config wiring from step 2 for all `reward.*` overrides to affect simulation behavior.
- Default runner command is:
- `python automation/run_pair_runner.py --config {config} --custom-env dial_mpc.envs.unitree_go2_env`
- You can replace it with `--runner-cmd` in both `run_single_experiment.py` and `run_sweep.py`.
- Standard trial artifacts are always written to each `exp_*` folder.
