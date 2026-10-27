import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 🔹 Load the dataset
train_data = pd.read_csv(r'c:\Users\user\Desktop\Train\Train_Data.csv')
test_data = pd.read_csv(r'c:\Users\user\Desktop\Test\Test_Data.csv')

# 🔹 Separate target variable
y_train = train_data["SalePrice"]
X_train = train_data.drop(columns=["SalePrice"])
X_test = test_data.copy()

# 🔹 Identify categorical & numerical features
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(exclude=['object']).columns

# 🔹 Handling missing values and transformations
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_features),
    ("cat", cat_pipeline, categorical_features)
])

# 🔹 Splitting data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 🔹 Apply preprocessing
X_train_prepared = preprocessor.fit_transform(X_train)
X_val_prepared = preprocessor.transform(X_val)
X_test_prepared = preprocessor.transform(X_test)

# 🔹 Train the Regression Model
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
model.fit(X_train_prepared, y_train)

# 🔹 Evaluate performance using RMSE
y_pred = model.predict(X_val_prepared)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")

# 🔹 PREDICT FOR ONE HOUSE
def predict_single_house_price(house_data):
    """
    Predict the price for a single house
    
    Parameters:
    house_data (dict or pandas Series): Feature values for one house
    
    Returns:
    float: Predicted house price
    """
    # Convert to DataFrame with one row
    house_df = pd.DataFrame([house_data])
    
    # Apply the same preprocessing
    house_prepared = preprocessor.transform(house_df)
    
    # Make prediction
    predicted_price = model.predict(house_prepared)[0]
    
    return predicted_price

# 🔹 EXAMPLE: Predict price for one specific house from the test set
# Let's take the first house from the test data as an example
single_house = test_data.iloc[0]  # Get the first house

print("\n" + "="*50)
print("PREDICTING PRICE FOR SINGLE HOUSE")
print("="*50)

# Display the house features
print("House Features:")
print(f"ID: {single_house['Id']}")
print("Some key features:")
for feature in ['LotArea', 'OverallQual', 'GrLivArea', 'BedroomAbvGr', 'FullBath']:
    if feature in single_house:
        print(f"  {feature}: {single_house[feature]}")

# Predict the price
predicted_price = predict_single_house_price(single_house)
print(f"\n🏠 PREDICTED HOUSE PRICE: ${predicted_price:,.2f}")

# 🔹 OPTIONAL: Create a function to input custom house data
def predict_custom_house():
    """
    Function to predict price for a custom house by specifying feature values
    """
    print("\n" + "="*50)
    print("PREDICT PRICE FOR CUSTOM HOUSE")
    print("="*50)
    
    # Example: Create a custom house with some key features
    # You would need to specify all the features that your model expects
    custom_house = {}
    
    # Get some key features from user (you can expand this)
    print("Enter house features (press Enter for default/missing values):")
    
    # Numerical features example
    numerical_examples = {
        'LotArea': 8450,
        'OverallQual': 7,
        'GrLivArea': 1710,
        'BedroomAbvGr': 3,
        'FullBath': 2,
        'YearBuilt': 2003
    }
    
    for feature, default_value in numerical_examples.items():
        try:
            value = input(f"{feature} (default: {default_value}): ")
            custom_house[feature] = float(value) if value else default_value
        except:
            custom_house[feature] = default_value
    
    # For categorical features, you'd need to specify them too
    # For simplicity, we'll use the first test house and modify numerical features
    base_house = test_data.iloc[0].copy()
    for feature, value in custom_house.items():
        if feature in base_house:
            base_house[feature] = value
    
    predicted_price = predict_single_house_price(base_house)
    print(f"\n🏠 PREDICTED PRICE FOR CUSTOM HOUSE: ${predicted_price:,.2f}")
    
    return predicted_price

# 🔹 Uncomment the line below to try custom house prediction
# predict_custom_house()

# 🔹 Predict prices for multiple individual houses
print("\n" + "="*50)
print("PREDICTIONS FOR FIRST 5 HOUSES IN TEST SET")
print("="*50)

for i in range(min(5, len(test_data))):
    house = test_data.iloc[i]
    predicted_price = predict_single_house_price(house)
    print(f"House {i+1} (ID: {house['Id']}): ${predicted_price:,.2f}")

# 🔹 Original functionality: Still save predictions for all test houses
test_preds = model.predict(X_test_prepared)
submission = pd.DataFrame({"Id": test_data["Id"], "SalePrice": test_preds})
submission.to_csv("HOUSEPREDICTION.csv", index=False)

print("\n✅ Submission file 'HOUSEPREDICTION.csv' has been saved successfully!")
print("✅ Single house prediction functionality added!")