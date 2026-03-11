import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib, os

# Load data
df = pd.read_csv("data/creditcard.csv")

# Feature engineering
df["hour"] = df.index % 24
df["merchant_risk"] = df["Amount"] / df["Amount"].max()
df["txn_count"] = np.random.randint(1,10,len(df))
df["device_trust"] = np.where(df["Class"]==1,0.4,0.8)
df["location_change"] = np.random.binomial(1,0.3,len(df))
df["international"] = np.random.binomial(1,0.2,len(df))

X = df[[
 "Amount","hour","merchant_risk",
 "txn_count","device_trust",
 "location_change","international"
]]

y = df["Class"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Balance
X,y = SMOTE().fit_resample(X,y)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Base model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5
)

# Probability calibration
model = CalibratedClassifierCV(rf, method="sigmoid")
model.fit(X_train, y_train)

os.makedirs("models",exist_ok=True)
joblib.dump(model,"models/fraud_model.pkl")
joblib.dump(scaler,"models/scaler.pkl")

print("Calibrated model trained & saved")
