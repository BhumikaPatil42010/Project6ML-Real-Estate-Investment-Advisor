# =========================
# FINAL CLEAN TRAIN.PY (NO ERRORS + NO LEAKAGE)
# =========================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib


# =========================
# LOAD DATA
# =========================

base_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_dir, 'data', 'D:/RealEstateProject/data/cleaned_data.csv')

df = pd.read_csv(data_path)

print("Data Loaded Successfully!")


# =========================
# REMOVE STRING COLUMNS
# =========================

df = df.select_dtypes(exclude=['object'])
print("Removed all string columns")


# =========================
# 🚨 REMOVE LEAKAGE FEATURES
# =========================

leak_cols = [
    'Price_Per_SqFt'   # highly correlated with price
]

df.drop(columns=[col for col in leak_cols if col in df.columns], inplace=True)

print("Removed leakage features")


# =========================
# FEATURES & TARGETS
# =========================

X = df.drop(['Price_in_Lakhs', 'Good_Investment'], axis=1)

y_class = df['Good_Investment']
y_reg = df['Price_in_Lakhs']

X = X.astype(float)

print("Features ready:", X.shape)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)


# =========================
# CLASSIFICATION MODEL
# =========================

clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train_c)

y_pred_c = clf.predict(X_test)

acc = accuracy_score(y_test_c, y_pred_c)
f1 = f1_score(y_test_c, y_pred_c)

print("\nClassification Results:")
print("Accuracy:", acc)
print("F1 Score:", f1)


# =========================
# REGRESSION MODEL
# =========================

reg = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
reg.fit(X_train, y_train_r)

y_pred_r = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("\nRegression Results:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)


# =========================
# SAVE MODELS
# =========================

model_path = os.path.join(base_dir, 'models')
os.makedirs(model_path, exist_ok=True)

joblib.dump(clf, os.path.join(model_path, 'classifier.pkl'))
joblib.dump(reg, os.path.join(model_path, 'regressor.pkl'))

print("\nModels saved successfully!")
print("\nTRAINING COMPLETED SUCCESSFULLY!")

