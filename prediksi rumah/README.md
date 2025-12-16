# California Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Library-Pandas-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Results & Evaluation](#results--evaluation)
- [Installation & Usage](#installation--usage)
- [Dependencies](#dependencies)

---

## Project Overview
This project presents an analysis and predictive model for median house values within California districts, leveraging various census data features. It illustrates a complete data science workflow, encompassing data acquisition, cleaning, feature engineering, model development (both custom implementations and library-based), and thorough evaluation.

## Problem Statement
The primary objective is to develop a regression model capable of accurately estimating `MedHouseVal` (Median House Value). This estimation relies on independent variables such as median income, house age, average number of rooms, and geographical coordinates. Addressing this classic regression challenge involves robust outlier handling, comprehensive understanding of feature correlations, and strategic selection of model complexity.

## Dataset
The **California Housing Dataset**, sourced from the `scikit-learn` library, is utilized for this project.

**Features:**
*   **MedInc:** Median income in a given block group (expressed in tens of thousands of US dollars).
*   **HouseAge:** Median age of houses within a block group.
*   **AveRooms:** Average number of rooms per household.
*   **AveBedrms:** Average number of bedrooms per household.
*   **Population:** Total population of the block group.
*   **AveOccup:** Average number of household members.
*   **Latitude:** Block group latitude coordinate.
*   **Longitude:** Block group longitude coordinate.

**Target Variable:**
*   **MedHouseVal:** Median house value for California districts (expressed in hundreds of thousands of US dollars).

---

## Methodology

### Exploratory Data Analysis (EDA)
*   **Distribution Analysis:** Distributions of the target variable and features were examined using histograms and box plots to identify potential skewness and the presence of outliers.
*   **Correlation Analysis:** A heatmap was generated to visualize the correlative relationships between features and with the target variable. Median income (`MedInc`) was identified as having the strongest correlation with house value.
*   **Geospatial Analysis:** The spatial distribution of median house prices across California was visualized using Latitude and Longitude coordinates, highlighting regions with higher property values, particularly along coastal areas.

### Preprocessing & Feature Engineering
1.  **Outlier Handling:** Outliers were identified using the Interquartile Range (IQR) method. Data points where `MedHouseVal` was capped at 5.0 were removed to enhance the model's generalization capabilities.
2.  **Feature Engineering:** New features were engineered to capture more intricate relationships within the data:
    *   `RoomPerPerson`: Calculated as `AveRooms / AveOccup`.
    *   `BedroomRatio`: Calculated as `AveBedrms / AveRooms`.
    *   `PopulationDensity`: Calculated as `Population / (AveOccup + 1)`.
3.  **Data Splitting:** The dataset was partitioned into an 80% training set and a 20% testing set to ensure proper model validation.
4.  **Feature Scaling:** `StandardScaler` was applied to normalize the feature set. This step is crucial for optimizing the performance and stability of Gradient Descent-based and regularized models.

---

## Modeling Approach

The project explored and compared three primary modeling strategies:

1.  **Linear Regression (Custom Implementation):**
    *   Developed a linear regression model from first principles, utilizing the **Normal Equation** for a closed-form solution.
    *   Further implemented linear regression with **Gradient Descent** optimization to illustrate the iterative learning process.
    
2.  **Linear Regression (Scikit-Learn Implementation):**
    *   The `scikit-learn` library's `LinearRegression` model was employed as a benchmark.
    *   The results from this implementation were compared with the custom-built model to validate the accuracy of the custom algorithms.

3.  **Regularized Regression:**
    *   **Ridge Regression (L2 Regularization):** Investigated different alpha parameter values to manage model complexity and mitigate overfitting.
    *   **Lasso Regression (L1 Regularization):** Applied for both regularization and intrinsic feature selection capabilities.

---

## Results & Evaluation

Model performance was rigorously assessed using standard regression metrics: **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, **R-squared (R²)**, and **Mean Absolute Percentage Error (MAPE)**.

### Key Performance Metrics (Test Set):
*   **R² Score:** Approximately 0.62, indicating that roughly 62% of the variance in median house values can be explained by the model.
*   **RMSE:** Approximately 0.60, representing the typical error magnitude in house value predictions (in hundreds of thousands of dollars).
*   **MAE:** Approximately 0.44, indicating the average absolute difference between predicted and actual house values (in hundreds of thousands of dollars).

### Residual Analysis:
*   **Residuals vs. Fitted Values Plot:** Used to check for homoscedasticity (constant variance of errors).
*   **Q-Q Plot:** Employed to visually assess the normality of the residuals.
*   **Durbin-Watson Test:** The statistic was approximately 1.96, suggesting the absence of significant autocorrelation in the residuals.

### Feature Importance:
Based on the model coefficients, the following features demonstrated the most significant influence on house prices:
1.  **Median Income (MedInc):** Exhibited a strong positive correlation.
2.  **Location (Latitude/Longitude):** Showed a notable inverse correlation, which can be attributed to the specific geographic distribution of high-value properties within California.
3.  **Rooms Per Person (RoomPerPerson):** Displayed a positive relationship with median house value.

---

## Installation & Usage

To set up and run this project:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd prediksi-rumah
    ```

2.  **Install dependencies:**
    Ensure Python 3.x is installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    (A `requirements.txt` file is recommended to be created for dependency management.)

3.  **Execute the Notebook:**
    Launch Jupyter Notebook and open the `main.ipynb` file:
    ```bash
    jupyter notebook main.ipynb
    ```

## Dependencies
The project relies on the following Python libraries:
*   Python (version 3.8 or higher recommended)
*   NumPy
*   Pandas
*   Matplotlib
*   Seaborn
*   Scikit-Learn
*   SciPy
*   Statsmodels

---
This README provides a professional and detailed overview of the California Housing Price Prediction project.