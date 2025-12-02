# MLTor

## Project Overview

This repository contains data, feature extraction code, and evaluation scripts for Website Fingerprinting (WF) under both **Closed-World** and **Open-World** scenarios.

The primary goals of this project are:

  - To provide a structured dataset of monitored and unmonitored traffic traces.
  - To extract handcrafted traffic features (Full / Robust / Basic).
  - To evaluate the robustness of traffic classification under various WF defense settings (split, padding, domain shift).

-----

## Dataset

**Download Link:** [Google Drive - MLTor Dataset](https://drive.google.com/drive/folders/1P4UgB3u28WC2NlkK1zNwzl-Txm8D0ISW?usp=sharing)

We provide serialized datasets for both closed-world and open-world experiments. Each dataset contains **monitored** (known sites) and **unmonitored** (unknown sites) traffic traces.

### File Structure

Please download the files from the link above and organize them into a `dataset` directory as follows:

  - `dataset/openworld/mon_standard.pkl`
  - `dataset/closeworld/unmon_standard10.pkl`
  - `dataset/closeworld/unmon_standard10_3000.pkl`

After downloading and organizing the datasets, **you must update the file paths** in the notebooks to match your local environment.

In each notebook (`Closed.ipynb`, `Open_binary.ipynb`, `Open_multi.ipynb`), locate the **CONFIGURATION** section at the top and modify the `PATHS` dictionary:

```python
PATHS = {
    # Update these paths to match your local file location
    "mon": "./your/local/path/dataset/openworld/mon_standard.pkl",
    "unmon": "./your/local/path/dataset/closeworld/unmon_standard10_3000.pkl"
}
```

*Ensure that the paths correctly point to where you saved the `.pkl` files on your machine.*

### File Descriptions

All datasets are serialized using the Python `pickle` format.

| File | Description |
| :--- | :--- |
| `mon_standard.pkl` | Monitored traffic from 95 websites (≈19k traces) |
| `unmon_standard10.pkl` | Unmonitored traffic (≈10k traces) |
| `unmon_standard10_3000.pkl` | Reduced unmonitored subset (≈3k traces) |

-----

## Experiment 1: Closed-World

**Goal**: Classify traffic traces into one of 95 monitored websites (Classes 0-94).

**Key Metrics**: Accuracy, Macro-F1.

### 1\. How to Run
You can reproduce the experimental results by running the main Jupyter Notebook. 
```python
Closed.ipynb
```

**Configuration**:
Locate the **Configuration** section at the beginning of the notebook and ensure the `SCENARIO` variable is set as follows:

```python
SCENARIO = 'closed'
```

*Simply run the entire notebook after setting this variable.*

### 2\. Execution Pipeline

1.  **Data Loading**: Automatically loads monitored traffic data. (*Note* Unmonitored data is skipped to optimize memory).
2.  **Model Optimization (RF vs XGB)**:
      * Compares Random Forest and XGBoost under various correlation thresholds (e.g., 0.95, 0.99).
      * Automatically selects the best model based on **Accuracy**.
3.  **Feature Selection**: Applies `EnhancedPreProcessor` to remove highly correlated features using the optimal threshold.
4.  **Final Evaluation**: Outputs the **Best Model, Optimal Threshold, Accuracy**, and **Macro-F1** score.

### 3\. Expected Output
The console will display the model selection process and final results:
```text
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

## Experiment 2: Open-World (Binary)

**Goal**: Determine whether a given web traffic trace corresponds to a **Monitored website (Known)** or an **Unmonitored website (Unknown)**.
This is achieved by training a model on 95 Monitored classes and using **Open-Set Rejection** (Confidence Scoring) to reject Unknown samples.

**Label Assignment**:

  * **Monitored Websites**: `1` (Internally 0-94 for multi-class training)
  * **Unmonitored Websites**: `-1`

**Key Metrics**:

  * **ROC-AUC**: Primary criterion for model selection (Measures overall discriminative power).
  * **TPR (Recall)**: Measures the rate of correctly identifying Known/Monitored traffic.
  * **FPR**: Measures the rate of incorrectly classifying Unknown/Unmonitored traffic as Known (Lower is better).
  * **Precision**: Measures the rate of correctly rejecting Unknown/Unmonitored traffic (TNR = 1 - FPR).

### 1\. How to Run
You can reproduce the experimental results by running the main Jupyter Notebook. 
```python
Open_binary.ipynb
```

**Configuration**:
Locate the **Configuration** section at the beginning of the notebook and ensure the `SCENARIO` variable is set as follows:
```python
SCENARIO = 'open_binary'
```

### 2\. Execution Pipeline
1. **Data Loading**: Automatically loads monitored traffic data and unmonitored traffic data.

2.  **Binary Label Construction**:
      * Monitored traffic → `1`
      * Unmonitored traffic (value defined in `CURRENT_CONFIG['unmon_label']`) → `-1`

3.  **Train/Test Split**:
      * The full dataset (X, y_binary) is split into training (X_train, y_train) and testing (X_test, y_test) sets using stratified sampling `stratify=y_binary` to preserve class balance.
      * `test_size` and `random_state` from `CURRENT_CONFIG`
        
4.  **Nested Evaluation Loop**:
    A results list (`results_binary`) is created and filled with results for every combination of:

      * Correlation Thresholds — from `CURRENT_CONFIG['corr_th']`
      * Confidence Threshold Percentiles — from `CURRENT_CONFIG['threshold_percentiles']`
      * Models — Random Forest (RF) and XGBoost (XGB)
      * This produces a full grid of experiments.

5.  **Preprocessing (per Correlation Threshold)**:
    For each `corr_th` value:
      * `EnhancedPreprocessor(correlation_threshold=corr_th)` is created.
      * `X_train` → `fit_transform`
      * `X_test` → `transform`
      * This removes highly correlated features according to the specified threshold.

6.  **Model Training (Once per Correlation Threshold)**:
    * For each correlation threshold, the notebook trains:
      * Random Forest (RF)
        *  Trained on `X_train_prep` with binary labels `y_train` 
      * XGBoost (XGB)
        * Uses CustomLabelEncoder to convert the binary labels {-1, 1} into encoded form {0, 1} for XGBoost's `binary:logistic` objective.
        * Trained on preprocessed features and encoded labels.
        * Note: Each model is trained only once per correlation threshold; threshold tuning happens afterward.
       
7.  **Confidence-Based Threshold Tuning & Rejection**:
    * For each threshold percentile (`th_pct`):
      * Step A — Compute Confidence
          * RF: `rf_proba = rf_model.predict_proba(X_test_prep)`
          * XGB: `xgb_proba = xgb_model.predict_proba(X_test_prep)`
          * Confidence for each sample:
            ```python
            conf = np.max(proba, axis=1)
            ```
      * Step B — Compute Decision Threshold
           ```python
           threshold = np.percentile(conf, th_pct)
           ```
 
      * Step C — Apply Rejection Rule
         * If `confidence < threshold`, reassign prediction to `-1` (unmonitored).

     * This implements an open-world rejection mechanism: If the model is not confident enough, classify as unmonitored.

8.  **Final Model Selection**:
     * The final model is automatically selected based on the highest **ROC-AUC** score achieved across all configurations.
     * Outputs the **Best Model, Optimal Correlation Threshold, Optimal Threshold Percentile, ROC-AUC, TPR, FPR**, and **Precision**.

### 3\. Expected Output
The console will display the model selection process and final results:
```text
================================================================================
OPEN-WORLD BINARY MODEL SELECTION  (Primary: ROC-AUC)
================================================================================
 SELECTED MODEL: XGB
  • Corr Threshold: 1.0
  • Threshold Percentile: 3.0%
  • ROC-AUC: 0.9676
  • TPR:     0.9863
  • FPR:     0.2733
  • Precision: 0.9581
================================================================================
```

-----

## Experiment 3: Open-World (Multi-Class)

**Goal**:

1. Determine whether an incoming traffic trace belongs to a **Monitored website (Known, classes 0–94)** or an **Unmonitored website (Unknown, -1)**.
2. If the trace is accepted as Monitored, classify it into one of the **95 monitored website classes**.

**Key Metrics**:

 * **Binary Detection Metrics**
   * **ROC-AUC** — *Primary* model-selection metric
   * **Precision** — *Secondary* selection metric
   * **TPR** — True Positive Rate (Monitored correctly accepted)
   * **FPR** — False Positive Rate (Unmonitored incorrectly accepted)
   * **TNR** — True Negative Rate (Correct Unknown rejection)
   * **PR-AUC** — Precision–Recall AUC

 * **95-Class Identification Metrics**
   * Computed **only on samples that pass the open-set rejection**:
   * **Monitored Accuracy**
   * **Monitored Macro-F1**

 * **Overall Metrics (All Test Samples)**
   * **Overall Accuracy**
   * **Overall Macro-F1**


### 1\. How to Run

Run the notebook:
`Open_multi.ipynb`

**Configuration**:
Locate the **Configuration** section at the beginning of the notebook and ensure the `SCENARIO` variable is set as follows:
```python
SCENARIO = 'open_multi'
```

### 2\. Execution Pipeline
1.  **Data Loading**
     * The notebook loads:
       * **Monitored dataset** (95 classes)
       * **Unmonitored dataset** (label = `-1`)
       * **Important:**
          * Only Monitored samples are used for training.
          * Unmonitored samples are used **only for testing** → ensures true open-set evaluation.
     
2.  **Train/Test Split**

     ```python
     # Monitored split (train + test)
     X_mon_train, X_mon_test, y_mon_train, y_mon_test = train_test_split(..., stratify=y_mon)
     
     # Unmonitored split (test only)
     _, X_unmon_test, _, y_unmon_test = train_test_split(...)
     ```
     * **Combined test set:**
       ```python
       X_test_all = np.vstack([X_mon_test_prep, X_unmon_test_prep])
       y_test_all = np.concatenate([y_mon_test, y_unmon_test])
       ```
3.  **Nested Evaluation Loop**
     * For each:
       * **Correlation Threshold** ∈ `[1.0, 0.99, 0.98, 0.95, 0.9]`
       * **Threshold Percentile** ∈ `[10, 15, 20, 25, 30]`
       * **Model** ∈ `{RF, XGB}`
      the notebook performs a full evaluation. All results are appended to `results_open`.

4.  **Preprocessing — Per Correlation Threshold**
     * Your code applies the preprocessing as:
        ```python
        prep = EnhancedPreprocessor(correlation_threshold=corr_th)
        
        X_mon_train_prep = prep.fit_transform(X_mon_train_df)   # Fit only on monitored train
        X_mon_test_prep  = prep.transform(X_mon_test_df)
        X_unmon_test_prep= prep.transform(X_unmon_test_df)
        ```
     * The preprocessor performs:
       * Correlation-based feature pruning
       * StandardScaler normalization
       * Fit on **Monitored training only** (to avoid leakage)

5.  **Model Training**
     * Two models are trained **once per correlation threshold**.
       * **Random Forest**
         ```python
         rf_model = RandomForestClassifier(**CURRENT_CONFIG['rf'])
         rf_model.fit(X_mon_train_prep, y_mon_train)
         ```
       
       * **XGBoost**
         ```python
         le = CustomLabelEncoder()
         y_mon_train_enc = le.fit_transform(y_mon_train)
         
         xgb_model = XGBClassifier(
             objective='multi:softprob',
             num_class=len(le.mapper),
             **CURRENT_CONFIG['xgb']
         )
         xgb_model.fit(X_mon_train_prep, y_mon_train_enc)
         ```
     * Models are **never trained on unmonitored data**.

6.  **Precompute Probabilities (Optimization)**
     * Your updated code **precomputes probabilities only once**, improving speed:
       ```python
       rf_proba_all = rf_model.predict_proba(X_test_all)
       rf_proba_mon = rf_model.predict_proba(X_mon_test_prep)
       rf_conf_all = np.max(rf_proba_all, axis=1)
       rf_conf_mon = np.max(rf_proba_mon, axis=1)
       ```
     * Same for XGB.

7.  **Open-Set Rejection (Confidence Thresholding)**
      * For each percentile (`th_pct`):
        
        * **A. Compute Threshold from Monitored Test**
          ```python
          rf_threshold = np.percentile(rf_conf_mon, th_pct)
          ```
          
        * **B. Apply Rejection to All Test Samples**
          ```python
          rf_pred_all = rf_model.predict(X_test_all)
          rf_pred_all[rf_conf_all < rf_threshold] = -1   # Rejection rule
          ```
      * This implements: **Max-Softmax Open-Set Recognition (OSR)** Reject samples whose highest softmax probability is too low.

8.  **Evaluation**
     * Metrics computed using:
       ```python
       evaluate_open_world(...)
       ```
     
     * **Includes:**
       * **Binary Detection**
         TPR, FPR, TNR, Precision, ROC-AUC, PR-AUC
     
       * **Overall Metrics**
         Overall Accuracy, Overall Macro-F1
     
       * **Monitored 95-Class Metrics**
         Accuracy & Macro-F1 on correctly accepted monitored samples only.
     
     * Each result is appended to:
       ```
       results_open
       ```
9.  **Final Model Selection**
     * The best model is selected using:
       * **ROC-AUC** (Primary)
       * **Precision** (Secondary)
   
   * Matching your main code:
       ```python
       best = df_open.sort_values(['ROC_AUC', 'Precision'], ascending=[False, False]).iloc[0]
       ```

10. **95-Class Direct Classifier (Baseline)**

     *(Updated to match your new implementation)*
     
     * Your new code **no longer trains a 96-class classifier**.
     * Instead, it evaluates a proper **95-class monitored-only classifier**, trained only on monitored data.
     
     * This baseline:
       * Trains on: **Monitored train**
       * Tests on: **Monitored test + Unmonitored test**
       * Does **not** perform rejection
       * Shows performance degradation when unmonitored samples appear
     
     * **RF 95-class baseline:**
        ```python
        rf_95.fit(X_95_train_prep, y_95_train)
        rf_95_pred = rf_95.predict(X_95_test_prep)
        ```
      
     * **XGB 95-class baseline:**
        ```python
        xgb_95.fit(X_95_train_prep, y_95_train_enc)
        ```
     * Metrics stored in:
        ```
        results_96class → now renamed logically to 95-class baseline
        ```

### 3\. Expected Output
The console will display the model selection process and final results:
```text
================================================================================
OPEN-WORLD MULTI-CLASS MODEL SELECTION
Primary: ROC-AUC | Secondary: Precision
================================================================================
 SELECTED MODEL: RF
  • Corr Threshold: 0.99
  • Threshold Percentile: 15.0%
  • ROC-AUC:   0.8403
  • Precision: 0.9268
  • TPR:       0.8500
  • FPR:       0.4250
  • Overall Acc: 0.7541
================================================================================
```

-----

## Module: Break WF Defense (Defense Evaluation)

This module evaluates robustness against traffic analysis defenses.

### Dataset (Defense)

**Download Link:** [Google Drive - Defense Dataset](https://drive.google.com/drive/folders/1P4UgB3u28WC2NlkK1zNwzl-Txm8D0ISW?usp=sharing)

The dataset is stored in `.cell` format within compressed archives.
**Note:** Please extract the `.zip` files within your code execution environment.

| Type | File Name | Classes | Total Instances | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Monitored** | `mon_50.zip` | 50 (0–49) | 10,000 | Balanced (200/class) |
| **Unmonitored** | `unmon_5000.zip` | 1 (-1) | 5,000 | Single class |

### Instance Structure

Each instance contains multiple variations (defense simulations):

  * **Keys**: `['split_0', 'split_1', 'split_2', 'split_3', 'split_4', 'join']`
  * **Shape**: $(N, 3)$ where $N$ is packet count.
  * **Features**: `[Time, Direction, Signed Size]`

### Evaluation Scenarios

| Scenario | Description |
| :--- | :--- |
| **Closed-World** | Standard 50-class classification |
| **Open-Binary** | Detect whether a trace is from a monitored site |
| **Open-Multi** | Detect + classify monitored sites among unknown ones |


### Defense Simulation Scenarios

Traffic defenses are simulated through split-domain datasets and robust feature reduction:
  * **join → split:** models trained on normal traffic, tested on defended (domain-shifted) traces.
  * **split → split:** models trained and tested on defended traffic.

### How to Run
You can reproduce the experimental results by running the main Jupyter Notebook. 
```python
break_WF_defense.ipynb
```
This notebook covers feature extraction, preprocessing, and evaluation for all three scenarios (Closed, Open-Binary, Open-Multi).
You can run these steps sequentially in the provided notebook/script.

**Prerequisites:**
Ensure the dataset files (`mon_50.zip`, `unmon_5000.zip`) are located in the data directory as defined in the notebook.

### Common: Feature Extraction & Preprocessing

Before running any specific experiments, load the raw data and extract features.

---
### Experiment 1 : Closed-World

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
---

### Experiment 2 : Open-World (Binary)

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
---
### Experiment 3 : Open-World (Multi-class)
Evaluate model performance on open-set multi-class classification,
which jointly measures detection (monitored vs unmonitored) and class-level identification.

**Step 1: Run Scenarios (Auto-Threshold)**
Two primary setups are evaluated:
- Join → Split: Train on undefended (Join) traffic, test on defended (Split) traffic → tests domain-shift robustness.
- Split → Split: Train and test on defended traffic → tests in-defense adaptability.

``` python
# Run Open-Multi evaluation for all feature sets
df_multi_full   = run_open_multi_pipeline(mon_data_full,   unmon_data_full,   models_to_open_multi)
df_multi_robust = run_open_multi_pipeline(mon_data_robust, unmon_data_robust, models_to_open_multi)
df_multi_basic  = run_open_multi_pipeline(mon_data_basic,  unmon_data_basic,  models_to_open_multi)
```

**Step 2: Visualization**
- Compare detection and classification metrics across thresholds and feature sets.
```
# Plot Detection-F1 and Class-F1 by Threshold
plot_open_multi_f1(df_multi_full, df_multi_robust, df_multi_basic)

# Plot ROC-AUC trends by Feature Set
plot_open_multi_auc(df_multi_full, df_multi_robust, df_multi_basic)
```
---

## Reference Papers

  * **Deep Fingerprinting** (Sirinam et al., CCS 2018) — Baseline deep model.
  * **Subverting Website Fingerprinting Defenses with Robust Traffic Representation** (Shen et al., 2023) — Inspiration for TAM representations.

## Questions & Contact

Please address any questions to the authors/developers:
  * **Eunhyeon Kwon** (keh54110@gmail.com)
  * **Hyewon Kim** (hhongyeahh@gmail.com)
  * **Yoonhyung Park** (pyoon820@gmail.com)
  * **Jeongin Heo** (jeongin0822@gmail.com)

