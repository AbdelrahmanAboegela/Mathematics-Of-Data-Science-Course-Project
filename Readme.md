# Linear Regression with Gradient Descent - README

## **Project Overview**
This project implements linear regression using gradient descent from scratch and compares its performance to scikit-learn’s implementation. Two datasets were used:
1. **Boston Housing Dataset** - Predict housing prices.
2. **Combined Cycle Power Plant Dataset (CCPP)** - Predict energy output.

The project also includes hyperparameter tuning, visualizations, and a discussion of findings and limitations.

---

## **Workflow**

### 1. **Dataset Preparation**

#### **Boston Housing Dataset**
- **Description**:
  - Predicts the median value of owner-occupied homes (`MEDV`) based on 13 features.
  - Collected from U.S. census data for the Boston area.
- **Columns**:
  | Feature         | Description                                               |
  |-----------------|-----------------------------------------------------------|
  | `CRIM`          | Per capita crime rate by town                             |
  | `ZN`            | Proportion of residential land zoned for lots >25,000 sq. ft |
  | `INDUS`         | Proportion of non-retail business acres per town          |
  | `CHAS`          | Charles River dummy variable (1 = tract bounds river; 0 = otherwise) |
  | `NOX`           | Nitric oxide concentration (parts per 10 million)         |
  | `RM`            | Average number of rooms per dwelling                      |
  | `AGE`           | Proportion of owner-occupied units built prior to 1940    |
  | `DIS`           | Weighted distances to five Boston employment centers      |
  | `RAD`           | Index of accessibility to radial highways                 |
  | `TAX`           | Full-value property-tax rate per $10,000                  |
  | `PTRATIO`       | Pupil-teacher ratio by town                               |
  | `B`             | 1000(Bk - 0.63)^2, where Bk is the proportion of Black population |
  | `LSTAT`         | % lower status of the population                          |
  | `MEDV`          | Median value of owner-occupied homes (in $1000s)          |

#### **Combined Cycle Power Plant (CCPP) Dataset**
- **Description**:
  - Predicts net hourly energy output (`PE`) from power plants.
  - Collected from a Combined Cycle Power Plant over six years (2006–2011).
- **Columns**:
  | Feature         | Description                                               |
  |-----------------|-----------------------------------------------------------|
  | `AT`            | Ambient Temperature (°C)                                  |
  | `V`             | Exhaust Vacuum (cm Hg)                                    |
  | `AP`            | Ambient Pressure (mbar)                                   |
  | `RH`            | Relative Humidity (%)                                     |
  | `PE`            | Net hourly electrical energy output (MW)                  |

---

### 2. **Exploratory Data Analysis (EDA)**
- **Goals**:
  - Understand data structure, identify missing values, and visualize relationships.
- **Steps**:
  - Checked for missing and duplicate data.
  - Visualized correlations using heatmaps.
  - Generated scatter plots for feature-target relationships.

---

### 3. **Data Preprocessing**
- **Goals**:
  - Prepare data for model training.
- **Steps**:
  - Handled missing values (if any) by imputing with column means.
  - Scaled features using standardization (mean = 0, standard deviation = 1) to improve gradient descent performance.

---

### 4. **Model Implementation**
#### **Custom Gradient Descent**:
- Implemented gradient descent to optimize weights and bias:
  - Predictions:
   ![Predictions Equation](https://latex.codecogs.com/png.latex?%5Chat%7By%7D%20%3D%20X%20%5Ccdot%20%5Ctheta%20%2B%20b)
  - Cost Function: 
    ![Cost Function Equation](https://latex.codecogs.com/png.latex?J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%20%5Csum%20%28%5Chat%7By%7D%20-%20y%29%5E2)
  - Gradients:
    - Weights: 
      ![Weight Gradient](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20X%5ET%20%28%5Chat%7By%7D%20-%20y%29)
    - Bias: 
      ![Bias Gradient](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20b%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum%20%28%5Chat%7By%7D%20-%20y%29)
  - Updates:
    - Weights: 
      ![Weight Update](https://latex.codecogs.com/png.latex?%5Ctheta%20%3A%3D%20%5Ctheta%20-%20%5Calpha%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%7D)
    - Bias: 
      ![Bias Update](https://latex.codecogs.com/png.latex?b%20%3A%3D%20b%20-%20%5Calpha%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20b%7D)

#### **Scikit-learn Linear Regression**:
- Trained the same datasets using scikit-learn’s `LinearRegression` for comparison.

---

### 5. **Hyperparameter Tuning**
- Experimented with different learning rates (\( \alpha \)) to observe their effects.
- Plotted cost function convergence for each learning rate.
- Determined optimal learning rates for both datasets:
  - Boston Housing: \( \alpha = 0.05 \)
  - CCPP: \( \alpha = 0.1 \)

---

### 6. **Evaluation**
- Evaluated models using Mean Squared Error (MSE) and R-squared (\( R^2 \)):
  - Boston Housing:
    - Optimal MSE: ~21.89
    - \( R^2 \): ~0.741
  - CCPP:
    - Optimal MSE: ~20.77
    - \( R^2 \): ~0.929
- Visualized cost function convergence to validate optimization.

---

## **Challenges**
- Selecting appropriate learning rates required trial and error.
- Ensuring numerical stability during training by implementing divergence safeguards.

---

## **Lessons Learned**
- Gradient descent is highly sensitive to hyperparameters like learning rate.
- Proper data preprocessing (e.g., feature scaling) significantly improves model performance.
- Manual implementation of algorithms enhances understanding of optimization principles.

---

## **How to Run the Project**
1. **Environment Setup**:
   - Install Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
2. **Run the Jupyter Notebook**:
   - Open the notebook file provided in this repository.
   - Follow the code cells step-by-step for implementation, visualizations, and results.
3. **Modify Hyperparameters**:
   - Adjust learning rates or number of iterations in the gradient descent implementation to observe their effects.

---

## **Future Work**
- Implement adaptive learning rates (e.g., Adam optimizer).
- Use mini-batch gradient descent for large datasets.
- Automate hyperparameter tuning using grid search or cross-validation.
