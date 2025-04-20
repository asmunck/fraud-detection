import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Caminho para o dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/aimldataset.csv')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/aimldataset_processed.csv')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.joblib')

df = pd.read_csv(DATA_PATH)

# Preenchimento de valores ausentes
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# Remover colunas irrelevantes
cols_to_drop = ['nameOrig', 'nameDest']
df.drop(columns=cols_to_drop, inplace=True)

# One-hot encoding para a coluna 'type'
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Padronizar colunas numéricas
num_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Salvar scaler e dataset processado
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
joblib.dump(scaler, SCALER_PATH)
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f'Pré-processamento completo. Dataset processado salvo em {PROCESSED_DATA_PATH}')
