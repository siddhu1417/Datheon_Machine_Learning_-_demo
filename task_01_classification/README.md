# Task 01: Iris Flower Classification Using Machine Learning

## 1. Project Overview

This project implements a supervised machine learning classification pipeline using the Iris dataset. It trains and compares two classification algorithms: **Logistic Regression** and **Decision Tree**.

The project includes data exploration, preprocessing, model training, performance evaluation, visualization, and automated tests.

## 2. Objectives

- Load and analyze the Iris dataset.
- Perform exploratory data analysis (EDA).
- Preprocess the data and split it into training and testing sets.
- Train Logistic Regression and Decision Tree models.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Generate charts, classification reports, and confusion matrices.
- Save trained models for future use.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Pytest

## 4. Prerequisites

Before running the project, install:

- Python 3.12 or a compatible version supported by the project dependencies
- Git
- Visual Studio Code or another code editor (optional)

Check your Python and Git installations:

```powershell
python --version
git --version
```

## 5. Step-by-Step Installation and Execution

### Step 1: Clone the GitHub Repository

Open PowerShell and run:

```powershell
git clone https://github.com/siddhu1417/Datheon_Machine_Learning_-_Data_Science_Internship.git
```

Enter the repository folder:

```powershell
cd Datheon_Machine_Learning_-_Data_Science_Internship
```

### Step 2: Enter the Task 01 Project Folder

```powershell
cd task_01_classification
```

All the remaining commands in this README should be run from this directory.

### Step 3: Create a Virtual Environment

Create an isolated Python environment for the project:

```powershell
py -m venv .venv
```

This step only needs to be performed once for the project environment. If the `.venv` folder already exists, skip this command.

### Step 4: Install the Required Libraries

Upgrade pip:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install the project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Wait for the installation to finish before proceeding.

### Step 5: Verify the Python Environment

Check the Python version:

```powershell
.\.venv\Scripts\python.exe --version
```

Check that the main libraries can be imported:

```powershell
.\.venv\Scripts\python.exe -c "import pandas, numpy, sklearn, scipy, matplotlib, seaborn, joblib; print('All required libraries imported successfully')"
```

### Step 6: Run the Machine Learning Project

Execute the main program:

```powershell
.\.venv\Scripts\python.exe main.py
```

The program runs the classification pipeline, including:

1. Loading the dataset.
2. Summarizing the dataset and checking its features.
3. Performing exploratory data analysis.
4. Splitting the data into training and testing sets.
5. Training Logistic Regression and Decision Tree models.
6. Calculating evaluation metrics.
7. Generating plots and evaluation reports.
8. Saving the trained models.

When the program finishes successfully, the generated output files will be available in the `outputs` directory.

### Step 7: Run the Automated Tests

Execute the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The tests check the dataset-loading, model-training, and preprocessing functionality.

A successful run should report that all three tests pass.

### Step 8: View the Generated Output Files

List all generated files:

```powershell
Get-ChildItem .\outputs -Recurse -File
```

Open the visualization folder:

```powershell
Invoke-Item .\outputs\plots
```

Open the evaluation-results folder:

```powershell
Invoke-Item .\outputs\results
```

The trained models are saved in:

```text
outputs/models/
```

## 6. Project Structure

```text
task_01_classification/
├── data/
│   ├── iris.csv
│   └── README.md
├── src/
│   ├── dataset.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── train.py
│   └── visualize.py
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── outputs/
│   ├── plots/
│   ├── results/
│   └── models/
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

The `outputs` directories and their contents may be created or updated when the project runs.

## 7. Output Description

### Visualizations — `outputs/plots/`

The project generates visualizations such as:

- Class distribution
- Correlation heatmap
- Feature distributions
- Pairplot
- Confusion matrices
- Model comparison chart
- Decision tree visualization
- Decision boundary plot

### Evaluation Results — `outputs/results/`

This folder contains:

- Dataset summary
- Classification reports
- Confusion matrix CSV files
- Model comparison results
- Dataset split metadata

### Trained Models — `outputs/models/`

The trained Logistic Regression and Decision Tree models are saved as `.joblib` files when the pipeline completes successfully.

## 8. Troubleshooting

**Error: `No module named 'src'`**

Make sure your terminal is inside `task_01_classification` before running `main.py` or the tests.

**Error: `requirements.txt` not found**

Make sure you entered the project directory correctly:

```powershell
cd task_01_classification
```

**Error: A required library is missing**

Install the dependencies again:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Error: SciPy DLL is blocked by a Windows Application Control policy**

This is a Windows security restriction, not necessarily a problem with the project code. Use an approved Python environment or contact your device administrator rather than disabling the security policy.

## 9. Author

**Siddhivinayak Patil**

Machine Learning and Data Science Internship — Task 01

## 10. Repository

GitHub: [Datheon Machine Learning & Data Science Internship](https://github.com/siddhu1417/Datheon_Machine_Learning_-_Data_Science_Internship)