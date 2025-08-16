# SL assignment

This project is part of Georgia Tech's CS7641: Machine Learning course. It explores supervised learning methods on two datasets: **Global Cancer Patients** and **Company Bankruptcy**, focusing on KNN, SVM, and Neural Networks.

## 🔧 Environment Setup

### 2. Set up Python environment

```bash
# Install Python 3.10 and required packages
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Verify Python version
python3.10 --version  # Should show Python 3.10.x

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 📈 Dataset Location
Datasets are located in the `/data` directory.

## 🚀 Running the Experiments

```bash
python main.py
```
Running main.py will generate all graphs and metrics present in the report.

## 📂 Results

- All graphs and visualizations are saved to the `/figures` directory
- Evaluation metrics and logs are saved in the `/results` directory

## 🧪 Tested Configuration
- Python 3.10.13
- TensorFlow 2.15.0
- scikit-learn 1.4.2
- pandas 2.2.2
- numpy 1.26.4
- matplotlib 3.8.4
- seaborn 0.13.2