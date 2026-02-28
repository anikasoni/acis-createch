import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('data/concrete_data.csv')

FEATURES = ['wc_ratio', 'cement_content', 'admixture_pct', 'avg_temp_C',
            'humidity_pct', 'demould_time_h', 'maturity_index', 'strength_at_demould']

X = df[FEATURES]
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    eval_metric='mlogloss', random_state=42
)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test),
      target_names=['Low', 'Medium', 'High']))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/risk_model.pkl')
print('✅ Model saved to models/risk_model.pkl')