import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Cargar dataset local desde data/
data_path = os.path.join("data", "mobile_train.csv")
df = pd.read_csv(data_path)

# La columna objetivo es 'price_range'
X = df.drop(columns=["price_range"])
y = df["price_range"]

# 2. División y escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)
print(f"Precisión del modelo: {accuracy:.2f}")

# 4. Guardar artefactos en la carpeta 'models/'
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("¡Modelo y escalador guardados con éxito!")
