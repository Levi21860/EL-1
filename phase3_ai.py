import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import joblib
file_path=r'C:\Users\SThanuj\Desktop\EL\archive\temphumid.csv'
data = pd.read_csv(file_path)
data['tiempo1'] = pd.to_datetime(data['tiempo1'], format='%d.%m.%Y %H:%M')
data['year'] = data['tiempo1'].dt.year
data['month'] = data['tiempo1'].dt.month
data['day'] = data['tiempo1'].dt.day
data['hour'] = data['tiempo1'].dt.hour
data['minute'] = data['tiempo1'].dt.minute


input_cols = ["soil1","temo","hum"]
output_cols = ["year", "month", "day", "hour", "minute"]


X = data[input_cols].values
y = data[output_cols].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if len(np.unique(y)) == 2:  
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
else:  
    xgb_model = xgb.XGBRegressor()

xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
model_path = r'C:\Users\SThanuj\Desktop\EL\phase3_model.joblib'
joblib.dump(xgb_model, model_path)
print(f"Model saved to {model_path}")


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
