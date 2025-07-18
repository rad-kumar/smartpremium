# insurance_premium_prediction.ipynb

# Step 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

# Step 2: Load dataset
df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 3: Initial inspection
print("Train Shape:", df.shape)
print(df.dtypes)
print(df.isnull().sum())

# Step 3.1: Correlation heatmap with target variable
numerical_df = df.select_dtypes(include=['int64', 'float64'])
corr_with_target = numerical_df.corr()[['Premium Amount']].sort_values(by='Premium Amount', ascending=False)

plt.figure(figsize=(8, 10))
sns.heatmap(corr_with_target, annot=True, cmap='coolwarm')
plt.title('Correlation with Premium Amount')
plt.show()

# Step 3.2: Histogram plots for numerical columns
num_cols = numerical_df.columns.drop('Premium Amount')
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Step 4: Preprocessing
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('Premium Amount')
categorical_features = df.select_dtypes(include='object').columns.tolist()

# Addressing skewness
numerical_features_log = ['Annual Income']
numerical_features_std = list(set(numerical_features) - set(numerical_features_log))

log_transformer = Pipeline([
    ("log", FunctionTransformer(np.log1p, validate=True)),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

std_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("lognum", log_transformer, numerical_features_log),
    ("stdnum", std_transformer, numerical_features_std),
    ("cat", cat_pipeline, categorical_features)
])

# Step 5: Define features and target
X = df.drop("Premium Amount", axis=1)
y = df["Premium Amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training with MLflow
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
}

best_model = None
best_score = -np.inf

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"Model: {name}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}\n")

        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipe, "model")

        if r2 > best_score:
            best_score = r2
            best_model = pipe

# Step 7: Save best model
joblib.dump(best_model, "best_insurance_model.pkl")

# Step 8: Predict on test set
test_predictions = best_model.predict(test_df)
output_df = test_df.copy()
output_df['Predicted Premium Amount'] = test_predictions
output_df.to_csv("predictions_on_test.csv", index=False)
print("✅ Test predictions saved to predictions_on_test.csv")
