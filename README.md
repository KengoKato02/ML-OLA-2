# Stroke Risk Prediction Model

## Overview
This project was based around exploring a binary stroke risk dataset which covers a risk percentage outcome aswell as a binary outcomes based on the percentage. We use a GBM Regression model which proved to be best fit, in order to predict the users likelihood of having a stroke based on inputs. Using a dataset of approximately 70,000 patients, we've managed to train our model to make predictions of stroke risk percentage with high accuracy (R² ≈ 0.98) based on your answers to our questionnaire. Please be aware that we do not claim these results to be a source of absolute truth and advice you to seek professional medical attention if you score a high percentage.

## Authors
- Wanesa Wintmiller
- Simona Tingleff Kardel
- Kengo Reimers Kato

Source: [Stroke Risk Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset/code)

## Application Installation and Usage

### Requirements
- Python 3.8+
- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn

### Setup
```bash
# Clone the repository
git clone [https://github.com/KengoKato02/ML-OLA-2]
cd ML-OLA-2

# Create and activate virtual environment
python -m venv ola-2-venv
source ola-2-venv/bin/activate  
```

### Notebook
Can be found under the /notebook dir and no need to install any dependencies as the notebook is configured to work out of the box. 

### Application
Need to install:
- fastapi
- jinja2
- pandas

Can be run from project root using ```uvicorn applications.api.app:app --reload```

```bash
pip install fastapi jinja2 pandas
uvicorn applications.api.app:app --reload
```


