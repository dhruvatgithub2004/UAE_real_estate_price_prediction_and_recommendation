# UAE Real Estate Analysis, Price Prediction, and Recommendation System

## Overview

This project focuses on analyzing real estate data from the UAE, building machine learning models to predict property prices, and developing a recommendation system for properties. The primary goal is to provide insightful analysis of the real estate market and assist users in finding suitable properties based on their specific queries.

## Features & Functionality

### 1. Data Processing and Cleaning
The project includes robust data processing steps to prepare raw real estate listings for analysis and modeling:

- **Data Loading and Initial Inspection**: Reads raw CSV data and displays initial information, including data types and non-null counts.
- **Handling Missing Values**: Drops rows containing NaN values to ensure data quality.
- **Location Extraction and Standardization**:
  - Converts `displayAddress` entries to lowercase.
  - Extracts Emirates (e.g., Dubai, Sharjah, Ajman, Abu Dhabi, Fujairah, Ras Al Khaimah, Umm Al Quwain) from `displayAddress` and assigns to a new `Emirate` column. Null `Emirate` values are filled with "abu dhabi".
  - Extracts granular location details (e.g., Business Bay, Jumeirah Beach Residence) and maps them.
- **Numerical Feature Conversion**: Converts `sizeMin` to numeric type. Converts `bedrooms` and `bathrooms` to numeric, mapping "studio" bedroom listings to "0".
- **Handling "Plus" Rooms**: Creates boolean flags (`has_plus_bedroom`, `has_plus_bathroom`) to identify listings with a '+' (e.g., "2BR+Study"). These flags increment the respective bedroom/bathroom counts by 1.
- **Total_rooms Calculation**: Computes a `Total_rooms` feature by summing numerical `bedrooms` and `bathrooms`.
- **Outlier Treatment**: Caps outliers in numerical features (`bathrooms`, `bedrooms`, `sizeMin`, `Total_rooms`) using the Interquartile Range (IQR) method, adjusting values beyond 1.5 times the IQR from the 25th and 75th percentiles.
- **Contextual Feature Engineering**:
  - Generates a boolean `Has sea view` column based on whether the property description contains the word "sea".
  - Calculates `Average price of location` by determining the mean price for each unique location and mapping it back to the dataset.
- **Column Selection**: Drops less relevant columns (`title`, `displayAddress`, `addedOn`, `description`, `type`, `priceDuration`, `price_per_sqft`) before model training to streamline the dataset.

### 2. Machine Learning for Price Prediction
The project builds and evaluates multiple regression models to predict real estate prices:

- **Feature Preprocessing for Models**: Uses a `ColumnTransformer` to apply different preprocessing steps:
  - Applies `OrdinalEncoder` to the `furnishing` column (NO, PARTLY, YES).
  - Uses `OneHotEncoder` for nominal categorical features like `Emirate` and `location`.
  - Applies `StandardScaler` to numerical features (`bathrooms`, `bedrooms`, `verified`, `sizeMin`, `has_plus_bedroom`, `has_plus_bathroom`, `Total_rooms`, `Has sea view`, `Average price of location`).
- **Model Training**: Splits the dataset into training and testing sets and trains the following regression models:
  - Linear Regression
  - Random Forest Regressor (`n_estimators=200`, `random_state=42`)
  - Gradient Boosting Regressor (`n_estimators=100`, `learning_rate=0.1`, `random_state=42`)
  - XGBoost Regressor (`n_estimators=100`, `learning_rate=0.1`, `random_state=42`)
- **Model Evaluation**: Assesses model performance using:
  - **R² Score**: Quantifies the proportion of variance in the dependent variable predictable from independent variables.
  - **Root Mean Squared Error (RMSE)**: Measures the average magnitude of errors. For the Random Forest model, the overall RMSE was AED 9,646,234.04.
  - **Mean Absolute Error (MAE)**: Measures the average magnitude of absolute errors. For the Random Forest model, the overall MAE was AED 2,242,934.
  - **Cross-Validation**: Employs 5-fold cross-validation across all models for robustness and generalizability.
  - **Performance by Price Band**: Analyzes RMSE and MAE across price segments (<2M, 2M–10M, >10M) to understand model accuracy variations.
- **Feature Importance**: Identifies the most influential features in predicting property prices using the `feature_importances_` attribute from the Random Forest model.

### 3. Property Recommendation System
A content-based recommendation system suggests properties similar to a user's query:

- **Description Preprocessing**: Cleans property descriptions by converting text to lowercase, removing HTML tags, URLs, punctuation, and stopwords.
- **Embedding Generation**: Transforms cleaned property descriptions into numerical vector embeddings using the `all-MiniLM-L6-V2` Sentence Transformer model.
- **Similarity Calculation**: Computes the embedding for a user query and calculates cosine similarity between the query embedding and property embeddings.
- **Recommendations**: Sorts properties by similarity scores in descending order and returns the top K most similar properties, including details like `price`, `bathrooms`, `bedrooms`, `Emirate`, `location`, `description`, and `furnishing`.
- **Amenity Categorization**: Categorizes amenities (e.g., core_building_amenities, lifestyle_community_amenities, location_linked_perks, investor_focused_addons) to enhance search and filtering.

## Data Used

The project uses a real estate dataset from `uae_real_estate_2024.csv`, initially containing 5058 entries. After cleaning and preprocessing, the refined dataset consists of 4696 entries with 18-20 columns. Key original columns include `title`, `displayAddress`, `bathrooms`, `bedrooms`, `addedOn`, `type`, `price`, `verified`, `priceDuration`, `sizeMin`, `furnishing`, and `description`. Engineered features include `Emirate`, `location`, `price_per_sqft`, `has_plus_bedroom`, `has_plus_bathroom`, `Total_rooms`, `Has sea view`, and `Average price of location`.

## Technologies and Libraries

The project is implemented in Python and uses the following libraries:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations and array operations.
- **matplotlib.pyplot** and **seaborn**: For data visualization (box plots, scatter plots, histograms, feature importance bar charts).
- **re** and **string**: For advanced string processing and text cleaning (e.g., regex for removing HTML tags/URLs, punctuation removal).
- **nltk.corpus.stopwords**: For removing common stopwords from text descriptions.
- **scikit-learn (sklearn)**: For machine learning tasks, including:
  - `model_selection`: For `train_test_split` and `cross_val_score`.
  - `compose.ColumnTransformer`: For preprocessing pipelines.
  - `preprocessing`: For `OneHotEncoder`, `StandardScaler`, and `OrdinalEncoder`.
  - `linear_model.LinearRegression`.
  - `ensemble`: For `RandomForestRegressor` and `GradientBoostingRegressor`.
  - `metrics`: For `r2_score`, `mean_squared_error`, `mean_absolute_error`, and `make_scorer`.
- **xgboost**: For the `XGBRegressor` model.
- **sentence_transformers**: For generating text embeddings with `SentenceTransformer`.
- **pickle**: For serializing and saving the trained machine learning pipeline.

## Results & Insights

The Random Forest Regressor showed the best performance among evaluated models for price prediction, achieving an R² score of approximately 0.718 on the test set and an average cross-validation R² score of 0.701. Error analysis reveals performance variations across price bands:

- **Listings < AED 2M**: RMSE of AED 350,764 and MAE of AED 208,127, indicating good precision.
- **Listings AED 2M–10M**: RMSE of AED 2,922,905 and MAE of AED 1,130,129, showing moderate accuracy.
- **Listings > AED 10M**: RMSE of AED 29,367,591 and MAE of AED 15,744,601, suggesting challenges with high-value properties.

This discrepancy highlights that high-value properties may be influenced by unique, less quantifiable factors. Feature importance analysis provides insights into key price determinants. The recommendation system effectively leverages semantic similarity to provide relevant property suggestions based on natural language queries.

## How to Run

To use this real estate analysis, prediction, and recommendation system, follow these steps:

1. **Obtain Dataset**: Ensure you have the `uae_real_estate_2024.csv` file.
2. **Set Up Environment**: Install required Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `sentence-transformers`, `nltk`, and `xgboost`.
3. **Perform Data Preprocessing**: Execute data cleaning and feature engineering scripts to transform raw data, handling missing values, normalizing locations, converting data types, creating features, and treating outliers.
4. **Train Price Prediction Model**: Train the chosen regression model (e.g., Random Forest) using the preprocessed data and `ColumnTransformer`. Save the trained model pipeline for future use.
5. **Evaluate Model**: Run evaluation scripts to assess performance using R² score, RMSE, and MAE, including analysis by price bands.
6. **Use Recommendation System**: Load or generate property embeddings from cleaned descriptions. Use the `recommend_estate` function with a text query to retrieve relevant property recommendations based on semantic similarity.

## Analogy

Imagine this project as a highly efficient digital real estate agent. Data cleaning and feature engineering are like the agent meticulously organizing every property detail—from room counts, including 'plus' spaces, to square footage and sea views. Price prediction models are the agent’s valuation skills, estimating market worth with higher accuracy for common properties and less for ultra-luxury ones with unique attributes. The recommendation system is the agent’s intuition, understanding your desires (e.g., "I want a quiet family home near a park") and suggesting listings that match the essence of your preferences.

## Link to the app
https://huggingface.co/spaces/dhruvdesai15/real_estate_price_prediction
