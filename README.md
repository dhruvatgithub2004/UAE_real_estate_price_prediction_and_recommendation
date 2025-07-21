UAE Real Estate Analysis, Price Prediction, and Recommendation System
Overview
This project focuses on analysing real estate data from the UAE, building machine learning models to predict property prices, and developing a recommendation system for properties. The primary goal is to provide insightful analysis of the real estate market and assist users in finding suitable properties based on their specific queries.
Features & Functionality
1. Data Processing and Cleaning
The project includes robust data processing steps to prepare raw real estate listings for analysis and modelling:
• Data Loading and Initial Inspection: Reads raw CSV data and displays initial information, including data types and non-null counts.
• Handling Missing Values: Rows containing NaN values are dropped to ensure data quality.
• Location Extraction and Standardisation:
    ◦ displayAddress entries are converted to lowercase.
    ◦ Emirates (e.g., Dubai, Sharjah, Ajman, Abu Dhabi, Fujairah, Ras Al Khaimah, Umm Al Quwain) are extracted from displayAddress and assigned to a new Emirate column. Null Emirate values are filled with "abu dhabi".
    ◦ More granular location details (e.g., Business Bay, Jumeirah Beach Residence) are also extracted and mapped.
• Numerical Feature Conversion: sizeMin is converted to a numeric type. bedrooms and bathrooms are converted to numeric, with "studio" bedroom listings mapped to "0".
• Handling "Plus" Rooms: Boolean flags (has_plus_bedroom, has_plus_bathroom) are created to identify if listings originally contained a '+' (e.g., "2BR+Study"). These flags are then used to increment the respective bedroom/bathroom counts by 1.
• Total_rooms Calculation: A Total_rooms feature is computed by summing the numerical bedrooms and bathrooms.
• Outlier Treatment: Outliers in numerical features (bathrooms, bedrooms, sizeMin, Total_rooms) are capped using the Interquartile Range (IQR) method. Values beyond 1.5 times the IQR from the 25th and 75th percentiles are adjusted to the respective upper or lower limits.
• Contextual Feature Engineering:
    ◦ A boolean Has sea view column is generated based on whether the property description contains the word "sea".
    ◦ Average price of location is calculated by determining the mean price for each unique location and mapping it back to the dataset.
• Column Selection: Less relevant columns such as title, displayAddress, addedOn, description, type, priceDuration, and price_per_sqft are dropped before model training to streamline the dataset.
2. Machine Learning for Price Prediction
The project builds and evaluates multiple regression models to predict real estate prices:
• Feature Preprocessing for Models: A ColumnTransformer is used to apply different preprocessing steps to various feature types:
    ◦ Ordinal Encoding is applied to the furnishing column (NO, PARTLY, YES).
    ◦ One-Hot Encoding is used for nominal categorical features such as Emirate and location.
    ◦ Standard Scaling is applied to numerical features, including bathrooms, bedrooms, verified, sizeMin, has_plus_bedroom, has_plus_bathroom, Total_rooms, Has sea view, and Average price of location.
• Model Training: The dataset is split into training and testing sets, and the following regression models are trained:
    ◦ Linear Regression.
    ◦ Random Forest Regressor (n_estimators=200, random_state=42).
    ◦ Gradient Boosting Regressor (n_estimators=100, learning_rate=0.1, random_state=42).
    ◦ XGBoost Regressor (n_estimators=100, learning_rate=0.1, random_state=42).
• Model Evaluation: Model performance is assessed using various metrics:
    ◦ R² Score: Quantifies the proportion of variance in the dependent variable that is predictable from the independent variables.
    ◦ Root Mean Squared Error (RMSE): Measures the average magnitude of the errors. For the Random Forest model, the overall RMSE was AED 9,646,234.04.
    ◦ Mean Absolute Error (MAE): Measures the average magnitude of the absolute errors. For the Random Forest model, the overall MAE was AED 2,242,934.
    ◦ Cross-Validation: 5-fold cross-validation is employed across all models to ensure robustness and generalisability.
    ◦ Performance by Price Band: RMSE and MAE are also analysed across different price segments (<2M, 2M–10M, >10M) to understand model accuracy variations.
• Feature Importance: The project identifies the most influential features in predicting property prices using the feature_importances_ attribute from the Random Forest model.
3. Property Recommendation System
A content-based recommendation system is developed to suggest properties similar to a user's query:
• Description Preprocessing: Property descriptions are cleaned by converting text to lowercase, removing HTML tags, URLs, and punctuation, and eliminating stopwords.
• Embedding Generation: Cleaned property descriptions are transformed into numerical vector embeddings using the all-MiniLM-L6-V2 Sentence Transformer model.
• Similarity Calculation: For a given user query, its embedding is computed, and cosine similarity is calculated between the query embedding and the embeddings of all properties in the dataset.
• Recommendations: Properties are sorted by their similarity scores in descending order, and the top K most similar properties are returned. The recommendations include details such as price, bathrooms, bedrooms, Emirate, location, description, and furnishing.
• Amenity Categorisation: Various amenities commonly found in listings are categorised to potentially enhance search and filtering (e.g., core_building_amenities, lifestyle_community_amenities, location_linked_perks, investor_focused_addons).
Data Used
The project primarily uses a real estate dataset loaded from uae_real_estate_2024.csv. This dataset initially contains 5058 entries. After data cleaning and preprocessing, the refined dataset consists of 4696 entries with 18-20 columns. Key original columns include title, displayAddress, bathrooms, bedrooms, addedOn, type, price, verified, priceDuration, sizeMin, furnishing, and description. Engineered features like Emirate, location, price_per_sqft, has_plus_bedroom, has_plus_bathroom, Total_rooms, Has sea view, and Average price of location are also added.
Technologies and Libraries
The project is implemented in Python and leverages the following key libraries:
• pandas: For comprehensive data manipulation and analysis.
• numpy: For numerical computations and array operations.
• matplotlib.pyplot and seaborn: For data visualisation, including box plots, scatter plots, histograms, and feature importance bar charts.
• re and string: For advanced string processing and text cleaning (e.g., regex for removing HTML tags/URLs, punctuation removal).
• nltk.corpus.stopwords: Used for removing common stopwords from text descriptions.
• scikit-learn (sklearn): The backbone for machine learning tasks, including:
    ◦ model_selection: For train_test_split and cross_val_score.
    ◦ compose.ColumnTransformer: For creating preprocessing pipelines.
    ◦ preprocessing: For OneHotEncoder, StandardScaler, and OrdinalEncoder.
    ◦ linear_model.LinearRegression.
    ◦ ensemble: For RandomForestRegressor and GradientBoostingRegressor.
    ◦ metrics: For r2_score, mean_squared_error, mean_absolute_error, and make_scorer.
• xgboost: For the XGBRegressor model.
• sentence_transformers: Specifically SentenceTransformer for generating high-quality text embeddings.
• pickle: For serialising and saving the trained machine learning pipeline.
Results & Insights
The Random Forest Regressor generally showed the best performance among the evaluated models for price prediction, achieving an R² score of approximately 0.718 on the test set and an average cross-validation R² score of 0.701. However, the detailed error analysis reveals that the model's accuracy varies significantly across different price bands:
• For listings under AED 2M, the RMSE was AED 350,764 and MAE was AED 208,127, indicating good precision.
• For listings between AED 2M and AED 10M, the RMSE was AED 2,922,905 and MAE was AED 1,130,129, showing moderate accuracy.
• For listings over AED 10M, the RMSE was AED 29,367,591 and MAE was AED 15,744,601, suggesting the model struggles more with very high-value properties.
This performance discrepancy highlights that high-value properties might be influenced by unique, less quantifiable factors not fully captured by the current dataset or model. The feature importance analysis provides valuable insights into which property attributes contribute most to price determination.
The recommendation system effectively leverages semantic similarity to provide relevant property suggestions based on natural language queries, enhancing the user experience by moving beyond simple keyword matching.
How to Run/Usage
To utilise this real estate analysis, prediction, and recommendation system, follow these general steps:
1. Obtain Dataset: Ensure you have the uae_real_estate_2024.csv file.
2. Set Up Environment: Install all required Python libraries, including pandas, numpy, scikit-learn, matplotlib, seaborn, sentence-transformers, nltk, and xgboost.
3. Perform Data Preprocessing: Execute the data cleaning and feature engineering scripts to transform the raw data into a suitable format for modelling. This involves handling missing values, normalising locations, converting data types, creating new features, and treating outliers.
4. Train Price Prediction Model: Train the chosen regression model (e.g., Random Forest) using the preprocessed data and the ColumnTransformer. The trained model pipeline can be saved for future use.
5. Evaluate Model: Run the evaluation scripts to assess the model's performance using metrics like R² score, RMSE, and MAE, including performance analysis by specific price bands.
6. Use Recommendation System: Either load the pre-computed property embeddings or generate them from the cleaned descriptions. Then, use the recommend_estate function by providing a text query to retrieve relevant property recommendations based on their semantic similarity to your search criteria.

--------------------------------------------------------------------------------
Analogy: Imagine this project as a highly efficient digital real estate agent. The data cleaning and feature engineering are like the agent meticulously organising every single detail about a property – from the number of rooms, including any 'plus' spaces, to the exact square footage, and even if it has a sea view. The price prediction models are the agent's precise valuation skills, capable of estimating a property's market worth. Although, just like a human expert, it might be more accurate for common properties and slightly less so for ultra-luxury ones with unique, harder-to-quantify attributes. Finally, the recommendation system is the agent's intuition, listening to your desires ("I want a quiet family home near a park") and instantly pulling up the perfect listings that truly match your preferences, not just by keywords, but by understanding the essence of what you're looking for.
