import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

# Carregar o dataset processado
data_path = os.path.join(os.path.dirname(__file__), '../data/aimldataset_processed.csv')
df = pd.read_csv(data_path)

# Separar features e alvo
y = df['isFraud']
X = df.drop(['isFraud'], axis=1)

# Dividir em treino e teste (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Iniciar experimento MLflow
mlflow.set_experiment('fraude-deteccao')

param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

with mlflow.start_run(run_name='RandomForest_RandomizedSearch'):
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    # Randomized Search
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist_rf,
        n_iter=10, 
        cv=3,    
        verbose=2,
        random_state=42,
        n_jobs=-1, 
        scoring='f1'
    )

    rf_random.fit(X_train, y_train)

    # Melhor modelo encontrado
    best_rf = rf_random.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    # Avaliar métricas com o melhor modelo
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f'Random Forest (Best Params) - AUC-ROC: {auc:.3f}')
    print(f'Random Forest (Best Params) - Precision: {precision:.3f}')
    print(f'Random Forest (Best Params) - Recall: {recall:.3f}')
    print(f'Random Forest (Best Params) - F1-score: {f1:.3f}')
    print('Random Forest (Best Params) - Confusion Matrix:')
    print(cm)
    print(f'Melhores parâmetros encontrados: {rf_random.best_params_}')

    # Logar métricas e modelo no MLflow
    mlflow.log_param('model', 'RandomForest_RandomizedSearch')
    mlflow.log_params(rf_random.best_params_)
    mlflow.log_metric('best_cv_f1_score', rf_random.best_score_) 
    mlflow.log_metric('AUC-ROC', auc)
    mlflow.log_metric('Precision', precision)
    mlflow.log_metric('Recall', recall)
    mlflow.log_metric('F1-score', f1)
    np.savetxt("confusion_matrix.txt", cm, fmt="%d")
    mlflow.log_artifact("confusion_matrix.txt")
    mlflow.sklearn.log_model(best_rf, 'random_forest_best_model')

print("Script de treinamento concluído.")
