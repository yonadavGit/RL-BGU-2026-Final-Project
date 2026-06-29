# RL-BGU-2026-Final-Project

This project extends AdaChain, a learned adaptive blockchain framework, to handle unreliable node reports. Instead of assuming direct access to the true workload state, we model architecture selection under uncertainty using a POMDP-style belief state, then compare it against the original AdaChain baseline and a naive report-trusting approach.

![Project image](portrait-beautiful-british-shorthair-cat-600nw-2625934295.webp)

## AdaChain Step Simulator

Open `ui/index.html` in a browser to step through a simplified AdaChain episode loop: observe workload state, score candidate architectures, select an action, measure throughput, and update experience.

## Bayesian AdaChain Simulation

Install dependencies in the repo-local Python environment:

```bash
pip install -r requirements.txt
```

Run the synthetic comparison between oracle AdaChain, naive report averaging, robust median aggregation, Bayesian multiple imputation, and Bayesian posterior-uncertainty features:

```bash
python experiments/run_experiment.py --episodes 160 --unreliable-fraction 0.25 --output-dir results/smoke
```

The command writes per-episode metrics, a summary CSV, and plots under the selected `results/` directory.

Train a reusable throughput emulator from the original AdaChain CSV traces:

```bash
python experiments/train_reward_emulator.py
```

The trained model is written to `models/adachain_throughput_emulator.joblib`, with validation metrics in `models/adachain_throughput_emulator_metrics.json`.

Use the trained emulator as the simulator reward source:

```bash
python experiments/run_experiment.py --reward-source emulator --episodes 300 --unreliable-fraction 0.25 --output-dir results/emulator_025_300
```

Choose a non-adversarial unreliable-report scenario:

```bash
python experiments/run_experiment.py --report-scenario biased_outliers --episodes 300 --unreliable-fraction 0.25 --output-dir results/biased_outliers_025_300
python experiments/run_experiment.py --report-scenario persistent_bias --episodes 300 --unreliable-fraction 0.25 --output-dir results/persistent_bias_025_300
python experiments/run_experiment.py --report-scenario heterogeneous_noise --episodes 300 --unreliable-fraction 0.25 --output-dir results/heterogeneous_noise_025_300
python experiments/run_experiment.py --report-scenario reliability_drift --episodes 300 --unreliable-fraction 0.25 --output-dir results/reliability_drift_025_300
python experiments/run_experiment.py --report-scenario stale_reports --episodes 300 --unreliable-fraction 0.25 --output-dir results/stale_reports_025_300
python experiments/run_experiment.py --report-scenario outlier_bursts --episodes 300 --unreliable-fraction 0.25 --output-dir results/outlier_bursts_025_300
python experiments/run_experiment.py --report-scenario mixed_unreliable --episodes 300 --unreliable-fraction 0.25 --output-dir results/mixed_unreliable_025_300
```

Generate the PDF report after running the scenario experiments:

```bash
python experiments/generate_report.py --output results/report/bayesian_adachain_report.pdf
```

The report explains the algorithm, data-generation scenarios, metrics, and plots in detail.

Run the simulation tests with:

```bash
python -m unittest discover -s tests -v
```
