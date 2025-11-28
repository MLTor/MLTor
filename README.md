# MLTor

# Project Overview

This repository contains data, feature extraction code, and evaluation scripts for website-fingerprinting (WF) under both closed-world and open-world scenarios.

The goals are:
- to provide a structured dataset of monitored/unmonitored traffic traces,
- to extract handcrafted traffic features (full / robust / basic),
- to evaluate robustness of traffic classification under various WF defense settings (split, padding, domain shift).

# Dataset
We provide serialized datasets for both closed-world and open-world experiments.
Each dataset contains monitored (known sites) and unmonitored (unknown sites) traffic traces.
## Dataset Files
| File                        | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `mon_standard.pkl`          | Monitored traffic from 95 websites (≈19k traces) |
| `unmon_standard10.pkl`      | Unmonitored traffic (≈10k traces)                |
| `unmon_standard10_3000.pkl` | Reduced unmonitored subset (≈3k traces)          |

All datasets are serialized with Python pickle format and can be loaded directly:
```
import pickle
X = pickle.load(open('mon_standard.pkl', 'rb'))
```





# WF Defense Evaluation
## Evaluation Scenarios
| Scenario         | Description                                          |
| ---------------- | ---------------------------------------------------- |
| Closed-World | Standard 95-class classification                     |
| Open-Binary  | Detect whether a trace is from a monitored site      |
| Open-Multi   | Detect + classify monitored sites among unknown ones |
## WF Defense Simulation
Traffic defenses are simulated through split-domain datasets and robust feature reduction:
- join → split: models trained on normal traffic, tested on defended (domain-shifted) traces.
- split → split: models trained and tested on defended traffic.
## Run Full Pipeline
```
python src/_______________.py
```
This runs:
- Threshold optimization (run_auto_threshold_pipeline)
- Fixed split evaluation (run_fixed_split_pipeline)
- Visualization

## Reference Papers
- Deep Fingerprinting (Sirinam et al., CCS 2018) — baseline deep model.
- Subverting Website Fingerprinting Defenses with Robust Traffic Representation (Shen et al., 2023) — feature inspiration (TAM representation).

## Question & Comments
Please address any questions or comments to the authors of the paper. The main developers of this code are:

