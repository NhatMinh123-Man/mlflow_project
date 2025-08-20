
# MLflow MLOps Project - Classification Example

## Project Overview
This project demonstrates how to apply **MLOps with MLflow** for a simple classification task.  
We use **scikit-learn's make_classification** to generate synthetic data, train a **RandomForestClassifier**, tune hyperparameters, compare results, and save the best model.  
Finally, we integrate the best model into a **Flask web application** for online predictions.

## Requirements
- Python 3.8+
- scikit-learn
- mlflow
- flask
- joblib
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
mlflow_project/
│── src/
│   └── train.py          # training script with hyperparameter tuning
│── app/
│   ├── app.py            # Flask web app
│   └── templates/index.html
│── model/
│   ├── best_model.pkl    # best model saved locally
│   ├── accuracy.txt      # best accuracy
│   └── meta.json         # metadata (n_features, etc.)
│── mlruns/               # MLflow logs
│── README.md             # project description
```

## Training Models
Run experiments with different hyperparameters:
```bash
python src/train.py --n_estimators 100 --max_depth 5
python src/train.py --n_estimators 200 --max_depth 6
python src/train.py --n_estimators 150 --max_depth None --n_samples 2000
```

MLflow will log parameters, metrics, and models.  
The script automatically saves the **best model** (highest accuracy) to `model/best_model.pkl`.

## Flask Web Application
Start the Flask app:
```bash
cd app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.  
- You can manually enter feature values.  
- Or click "Random Predict" to generate random input.  
- The page shows prediction (class 0/1), confidence, and fixed accuracy of the best model.

## Repository Link
Please replace `repo_link.txt` in submission with your GitHub repository link.

## Author
Developed as part of an MLOps course project.
