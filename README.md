# Asteroid Hazard Predictor

A machine learning project that predicts whether a Near Earth Object (NEO) is hazardous to Earth, using NASA's official asteroid dataset.

Built with **Python** and **scikit-learn** — and includes a full from-scratch implementation of logistic regression using gradient descent, verified against sklearn's output.

---



---

## Results

| Model | Test Accuracy |
|---|---|
| scikit-learn LogisticRegression | 95.20% |
| Custom implementation (from scratch) | 95.31% |

---

## Dataset

**Source:** [NASA Near Earth Objects dataset](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects) — hosted on Kaggle.

- 4,687 asteroid records
- 40 features (orbital parameters, velocity, miss distance, diameter estimates, etc.)
- Binary target: `Hazardous` (True / False)
- Class distribution: ~84% non-hazardous, ~16% hazardous

---

## Project Structure

```
asteroid-hazard-predictor/
├── AsteroidHazardPredictor.ipynb   # Main notebook
├── nasa.csv                        # Dataset
└── README.md
```

---

## What's Inside the Notebook

### 1. Data preprocessing
- Dropped non-informative columns (IDs, single-value columns like `Equinox`)
- Used a correlation heatmap to identify and remove highly correlated features (reduces multicollinearity)
- Applied `StandardScaler` — fitted on training data only to prevent data leakage

### 2. Exploratory data analysis
- Correlation heatmap across all numerical features
- Class distribution check

### 3. sklearn model
- `LogisticRegression` with scaled features
- 80/20 train-test split (`random_state=10`)

### 4. From-scratch implementation
Implemented the full logistic regression algorithm manually:
- **Sigmoid function:** `g(z) = 1 / (1 + e^(-z))`
- **Cost function:** binary cross-entropy loss averaged over all examples
- **Gradient computation:** vectorised partial derivatives for weights and bias
- **Gradient descent:** iterative parameter update over 100,000 iterations (`α = 0.01`)
- **Cost vs. iteration plot:** to verify convergence

---

## How to Run

### — Local

```bash
# Clone the repo
git clone https://github.com/Kumarce/asteroid-hazard-predictor.git
cd asteroid-hazard-predictor

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch notebook
jupyter notebook AsteroidHazardPredictor.ipynb
```

---

## Tech Stack

- Python 3.x
- NumPy
- pandas
- Matplotlib & Seaborn
- scikit-learn

---

## Key Concepts Demonstrated

- Binary classification
- Feature selection via correlation analysis
- Data preprocessing and scaling
- Logistic regression (library + from scratch)
- Gradient descent optimization
- Train/test split and model evaluation

---

## Author

Rohit Kumar
[GitHub](https://github.com/Kumarce)
