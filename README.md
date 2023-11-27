# Automated-Hyperparameter-Tuning-with Optuna for RandomForestRegressor

## Overview
This project demonstrates the automation of hyperparameter tuning for a RandomForestRegressor model using the Optuna optimization framework. The goal is to find the optimal hyperparameters that minimize mean squared error on the Diabetes dataset

## Getting Started
Prerequisites
Python 3.x

1. Install required libraries:
  'pip install scikit-learn optuna joblib'

2. Installation
Clone the repository:

3. Set up a virtual environment (optional but recommended):
  python -m venv venv
  source venv/bin/activate  # On Windows, use "venv\Scripts\activate"

4. Install dependencies:
pip install -r requirements.txt

5.Usage
Run the Jupyter Notebook:
jupyter notebook

6. Open the hyperparameter_tuning.ipynb notebook.

7. Follow the instructions in the notebook to load the Diabetes dataset, define the objective function, create an Optuna study, and optimize hyperparameters.

# Save the trained model:

8. import joblib

# Load the best hyperparameters from the study
best_params = study.best_params

# Create and train the final model
final_model = RandomForestRegressor(**best_params)
final_model.fit(X, y)

# Save the model
joblib.dump(final_model, 'random_forest_model.joblib')

## Results
The best hyperparameters obtained from the optimization process are {'n_estimators': 66, 'max_depth': 3}, resulting in a mean squared error of **2774.34** on the test set.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Optuna Documentation ( https://optuna.readthedocs.io/en/stable/ ) 
Scikit-learn Documentation ( https://scikit-learn.org/stable/index.html )
# Feel free to contribute and open issues if you have suggestions or encounter any problems.
