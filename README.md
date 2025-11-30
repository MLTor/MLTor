# MLTor

## Project Overview

This repository contains data, feature extraction code, and evaluation scripts for website-fingerprinting (WF) under both closed-world and open-world scenarios.

The goals are:
- To provide a structured dataset of monitored/unmonitored traffic traces.
- To extract handcrafted traffic features (full / robust / basic).
- To evaluate robustness of traffic classification under various WF defense settings (split, padding, domain shift).

---

## Dataset

*(TODO: <혜원> Dataset 파일 폴더 만들어서 추가 하기)*

We provide serialized datasets for both closed-world and open-world experiments. Each dataset contains monitored (known sites) and unmonitored (unknown sites) traffic traces.

### Dataset Files
All datasets are serialized using the Python `pickle` format.

| File | Description |
| :--- | :--- |
| `mon_standard.pkl` | Monitored traffic from 95 websites (≈19k traces) |
| `unmon_standard10.pkl` | Unmonitored traffic (≈10k traces) |
| `unmon_standard10_3000.pkl` | Reduced unmonitored subset (≈3k traces) |

All datasets are serialized with Python pickle format and can be loaded directly:
```python
import pickle
X = pickle.load(open('mon_standard.pkl', 'rb'))
````

-----

## Closed-World

**Goal**: Classify traffic traces into one of 95 monitored websites(0-94). 

**Key Metrics**: Accuracy, Macro-F1. 

### 1. How to Run
You can reproduce the experimental results by running the main Jupyter Notebook. 
```python
Closed.ipynb
```

**Note**: To execute the **Closed-World** scenario, locate the **Configuration** section at the beginning of the notebook and ensure the `SCENARIO` variable is set correctly.

```SCENARIO = 'closed'```

*Simply run the entire notebook after setting this variable.*

### 2. Execution Pipeline
When `SCENARIO = 'closed'`, the following pipeline is executed:

1) **Data Loading**: 
 - Automatically loads monitored traffic data.
 - *Note*: Unmonitored data loading is skipped to optimize memory usage.

2) **Model Optimization (RF vs XGB)**:
 - Compares Random Forest and XGBoost under various correlation thresholds (e.g., 0.95, 0.99).
 - Automatically selects the best model based on **Accuracy**.

3) **Feature Selection**: 
 - Applies `EnhancedPreProcessor` to remove highly correlated feature using the optimal threshold.

4) **Final Evalution**: 
 - Outputs the **Best Model, Optimal Threshold, Accuracy**, and **Macro-F1** score.

### 3. Expected Output
The console will display the model selection process and final results:
```
================================================================================
 CLOSED-WORLD MODEL SELECTION  (Primary Metric: Accuracy)
================================================================================

 SELECTED MODEL: Random Forest
  • Corr Threshold: 0.95
  • Accuracy: 0.8412
  • Macro F1: 0.8350
================================================================================
```

-----

## Open-World : Binary

*(Content to be updated)*

-----

## Open-World : Multi-class

*(Content to be updated)*

-----

# Break WF Defense

### Dataset Files

The dataset for defense evaluation is stored in `.cell` format. These files contain hierarchical structures mapping labels to specific traffic instances under various defense settings (e.g., split, join).
**Please note that these files are provided as compressed archives (`.zip`) and must be extracted within the code execution environment.**

| Type | File Name | Classes (Labels) | Total Instances | Distribution |
| :--- | :--- | :--- | :--- | :--- |
| **Monitored** | `mon_50.zip` (.cell) | **50** (0–49) | **10,000** | Balanced (200 per class) |
| **Unmonitored** | `unmon_5000.zip` (.cell) | **1** (-1) | **5,000** | Single class |
---
### Instance Structure & Defense Keys

Each instance within the dataset contains multiple variations of the traffic trace corresponding to different defense simulations or data splits.

  * **Available Keys per Instance:** `['split_0', 'split_1', 'split_2', 'split_3', 'split_4', 'join']`
  * **Data Shape:** $(N, 3)$
      * $N$: Number of packets (variable length per trace)
      * $3$: Feature columns (Time, Direction, Size)
---
### Feature Matrix Details

| Column Index | Feature Name | Description | Example Values |
| :--- | :--- | :--- | :--- |
| **Col 0** | **Time** | Relative timestamp of the packet | `0.0000`, `0.1270` |
| **Col 1** | **Direction** | Packet direction (-1: Incoming, 1: Outgoing) | `-1`, `1` |
| **Col 2** | **Signed Size** | Packet size combined with direction | `-512.0`, `512.0` |
---
### Evaluation Scenarios

| Scenario | Description |
| :--- | :--- |
| **Closed-World** | Standard 50-class classification |
| **Open-Binary** | Detect whether a trace is from a monitored site |
| **Open-Multi** | Detect + classify monitored sites among unknown ones |
---
### WF Defense Simulation

Traffic defenses are simulated through split-domain datasets and robust feature reduction:

  - **join → split:** models trained on normal traffic, tested on defended (domain-shifted) traces.
  - **split → split:** models trained and tested on defended traffic.

-----

## Run Full Pipeline

You can reproduce the experimental results by running the main Jupyter Notebook. 
```python
break_WF_defense.ipynb
```
This notebook covers feature extraction, preprocessing, and evaluation for all three scenarios (Closed, Open-Binary, Open-Multi).
You can run these steps sequentially in the provided notebook/script.

**Prerequisites:**
Ensure the dataset files (`mon_50.zip`, `unmon_5000.zip`) are located in the data directory as defined in the notebook.

### 1\. Common: Feature Extraction & Preprocessing

Before running any specific experiments, load the raw data and extract features.

### 2\. Experiment: Closed-World

Evaluate performance on 50 monitored sites using Multi-class Classification.

**Step 1: Create Train/Test Splits**
- Generate stratified train/test sets for both 'Join' (undefended) and 'Split' (defended) datasets.

```python
# Returns dictionaries containing X_train, X_test, y_train, y_test for each split
full_split_datasets, summary = create_split_trained_datasets(mon_data_full)
```

**Step 2: Run Evaluation Scenarios**
- Run both **Join-Trained** (Baseline) and **Split-Trained** (Robustness) scenarios.

```python
# Returns results for Scenario 1 (df_s1) and Scenario 2 (df_s2)
df_s1, df_s2 = run_all_scenarios(mon_data_full, full_split_datasets, models_to_closed)
```

**Step 3: Visualization**
- Compare performance across feature sets (Full vs. Robust vs. Basic).

```python
# Generate Bar plots and Line charts for Macro-F1 & Accuracy
plot_feature_set_comparison(df_s1_full, df_s2_full, ...)
plot_macro_f1_lines(df_join_all, df_split_all)
```

### 3\. Experiment: Open-World (Binary)

Evaluate the ability to distinguish Monitored vs. Unmonitored traffic.

**Step 1: Run Scenarios (Auto-Threshold)**
- Train the model and automatically find the optimal confidence threshold for detection.

```python
# Find optimal threshold (ROC-AUC, TPR, FPR, etc.)
df_auto_full = run_auto_threshold_binary_all(mon_data_full, unmon_data_full, models_to_open_binary)

# Run evaluation with fixed thresholds
df_fixed_full = run_fixed_threshold_binary_all(mon_data_full, unmon_data_full, models_to_open_binary, th_full)
```

**Step 2: Visualization**
- Visualize detection performance and trade-offs.

```python
# Plot F1-Score, ROC-AUC, and TPR vs. FPR curves
plot_open_binary_all(df_fixed_full, df_fixed_robust, df_fixed_basic)
  ```

### 4\. Experiment: Open-World (Multi-class)

*(Content to be updated)*

---

## Reference Papers

  - **Deep Fingerprinting** (Sirinam et al., CCS 2018) — baseline deep model.
  - **Subverting Website Fingerprinting Defenses with Robust Traffic Representation** (Shen et al., 2023) — feature inspiration (TAM representation).
---
## Question & Comments

Please address any questions or comments to the authors of the paper. The main developers of this code are:

  - **Yoonhyung Park** (pyoon820@gmail.com)
  - **Hyewon Kim** (hhongyeahh@gmail.com)
  - **Jeongin Heo** (jeongin0822@gmail.com)
  - *(TODO: \<팀원 이메일 추가하기\>)*

<!-- end list -->

```
```
