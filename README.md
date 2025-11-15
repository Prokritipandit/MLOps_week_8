# MLOps Assignment: Data Poisoning & Validation

This project investigates the impact of data poisoning attacks on a machine learning model's performance. We intentionally "poison" the Iris dataset at various levels and use MLflow to train, log, and validate the outcomes.

The goal is to visually demonstrate how data quality affects model accuracy and to discuss potential mitigation strategies. This experiment is self-contained and run from a Vertex AI Workbench instance.

## 🧰 Tech Stack

* **Cloud Environment:** Vertex AI Workbench
* **Experiment Tracking:** MLflow
* **Artifact Storage:** Google Cloud Storage (GCS)
* **Core Libraries:** Scikit-learn, Pandas, NumPy

---

## 📁 File Utility

This repository contains the following files:

* **`data_poisoning_experiment.ipynb`**: This is the main Jupyter Notebook for the assignment. It contains all the Python code to:
    1.  Load the clean Iris dataset.
    2.  Define a function to poison a percentage of the data with random, out-of-distribution numbers.
    3.  Run an MLflow experiment for each poison level (0%, 5%, 10%, 50%).
    4.  Train a Logistic Regression model on the (potentially) poisoned data.
    5.  Evaluate the model against a **clean, held-out test set**.
    6.  Log the `poison_percentage` (as a param) and `clean_test_accuracy` (as a metric) to MLflow.

* **`README.md`**: This file, explaining the project's objective, files, and key findings.

* **`.gitignore`**: This file tells Git to ignore local MLflow files (like `mlflow.db` and the `mlruns/` directory) that should not be committed to the repository.

---

## 🚀 How to Run the Experiment

1.  Launch the MLflow server in a dedicated terminal:
    ```bash
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root gs://[YOUR-GCS-ARTIFACT-BUCKET] \
        --host 0.0.0.0 \
        --port 5000
    ```
2.  Open and run all cells in the `data_poisoning_experiment.ipynb` notebook.
3.  Preview the MLflow UI (on port 5000) to see and compare the results.

---

## 📊 Validation Outcomes & Findings

The results are validated by comparing the experiment runs in the MLflow UI.

* **Observation:** There is a clear, inverse relationship between the `poison_percentage` and the model's performance.
* **Validation:**
    * The **0% poison** run acts as our baseline, achieving high accuracy (~97-100%) on the clean test set.
    * As the `poison_percentage` **increases** to 5% and 10%, the `clean_test_accuracy` **decreases** significantly.
    * At **50% poison**, the model's accuracy on clean data plummets, as it has learned the incorrect, noisy patterns from the poisoned data and can no longer generalize.



---

## 💡 Learnings from this Assignment

### How to Mitigate Poisoning Attacks

1.  **Input Validation:** This is the most effective defense. Implement automated checks (e.g., using Z-scores or Interquartile Range) to flag or reject new data that falls far outside the expected distribution (e.g., a petal length of 70 when the max should be 7).
2.  **Data Provenance:** Use data versioning (like DVC) and secure, access-controlled storage. This creates an auditable trail, so you know exactly where your data came from and when it last changed.
3.  **Adversarial Training:** If you can anticipate the *type* of attack, you can train your model on a small, labeled set of "bad" data to make it more robust and learn to ignore it.

### Data Quality vs. Data Quantity

* **The Key Insight:** As data **quality** *decreases*, the **quantity** of clean data required to achieve the same performance *increases exponentially*.
* **Analogy:** A model is trying to find a "signal" (the true pattern) in "noise" (bad data).
    * **High-Quality Data:** Strong signal, low noise. The model learns the pattern quickly with less data.
    * **Low-Quality Data:** Weak signal, high noise. The model learns the noise instead of the signal.
* **How Requirements Evolve:** To fix a model trained on low-quality data (like our 50% poisoned set), you can't just add a *little* more clean data. You must add a **massive** amount of verified, clean data to "drown out" the bad patterns and give the model a chance to re-learn the true signal.