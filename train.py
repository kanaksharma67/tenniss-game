# tennis_model_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import joblib

# ===============================
# 1. Load Data
# ===============================
df = pd.read_csv("dataset_enriched_for_lines.csv")

# ===============================
# 2. Basic Preprocessing
# ===============================
# Drop non-useful columns
drop_cols = ['winner_name', 'loser_name', 'tournament', 'ATP', 'Location', 'Comment']
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Drop rows with missing values (or you can fillna if needed)
df.dropna(inplace=True)

# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ===============================
# 3. Define Targets
# ===============================
# Winner Prediction (Classification)
y_class = df['winner'] if 'winner' in df.columns else df['moneyline_winner']
# Total Games Prediction (Regression)
y_reg = df['total_games']

# Features
X = df.drop(columns=['total_games', 'winner'] if 'winner' in df.columns else ['total_games', 'moneyline_winner'])

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# ===============================
# 4. Classification Models (Winner)
# ===============================
print("\n🏆 Training Classification Models for Winner Prediction...")

clf_models = {
    "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

clf_params = {
    "XGBoost": {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]},
    "RandomForest": {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
    "LogisticRegression": {'C': [0.1, 1, 10]}
}

best_clf = None
best_acc = 0

for name, model in clf_models.items():
    print(f"\n🔍 Grid Search for {name}...")
    grid = GridSearchCV(model, clf_params[name], cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train_c, y_train_c)
    y_pred = grid.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test_c, y_pred))
    if acc > best_acc:
        best_acc = acc
        best_clf = grid.best_estimator_

print(f"\n✅ Best Classification Model: {best_clf}")
joblib.dump(best_clf, "best_winner_model.pkl")

# ===============================
# 5. Regression Models (Total Games)
# ===============================
print("\n🎾 Training Regression Models for Total Games Prediction...")

reg_models = {
    "XGBoostRegressor": XGBRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42)
}

reg_params = {
    "XGBoostRegressor": {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]},
    "RandomForestRegressor": {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
}

best_reg = None
best_rmse = float("inf")

for name, model in reg_models.items():
    print(f"\n🔍 Grid Search for {name}...")
    grid = GridSearchCV(model, reg_params[name], cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
    grid.fit(X_train_r, y_train_r)
    y_pred = grid.predict(X_test_r)
    rmse = mean_squared_error(y_test_r, y_pred, squared=False)
    print(f"{name} RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_reg = grid.best_estimator_

print(f"\n✅ Best Regression Model: {best_reg}")
joblib.dump(best_reg, "best_total_games_model.pkl")

# ===============================
# 6. Predict Over/Under Line
# ===============================
OVER_UNDER_LINE = 22.5

def predict_match(features):
    features_scaled = scaler.transform([features])
    winner_pred = best_clf.predict(features_scaled)[0]
    total_games_pred = best_reg.predict(features_scaled)[0]
    over_under = "Over" if total_games_pred > OVER_UNDER_LINE else "Under"
    return winner_pred, total_games_pred, over_under

print("\n🚀 Models Trained and Saved! Use predict_match(features) to make predictions.")
