import mlflow
import joblib
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

BEST_RUN_ID = '1ace418a929a4116867671ae8a50fc71'
MODEL_ARTIFACT_PATH = 'random_forest_best_model'
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.joblib')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/aimldataset_processed.csv')

def load_artifacts(run_id, model_path, scaler_path):
    try:
        model_uri = f"runs:/{run_id}/{model_path}"
        model = mlflow.sklearn.load_model(model_uri)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Erro ao carregar artefatos: {e}")
        return None, None

def preprocess_input(data, scaler, expected_columns):
    df_input = data.copy()
    df_input = pd.get_dummies(df_input, columns=['type'], drop_first=True)
    df_input = df_input.reindex(columns=expected_columns, fill_value=0)
    num_cols = scaler.feature_names_in_
    cols_to_scale = [col for col in num_cols if col in df_input.columns and pd.api.types.is_numeric_dtype(df_input[col])]
    if cols_to_scale:
        df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])
    return df_input

def predict_fraud(input_data, model, scaler, expected_columns):
    if model is None or scaler is None:
        print("Modelo ou scaler não carregado. Abortando previsão.")
        return None
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        print("Formato de entrada inválido. Use dict ou DataFrame.")
        return None
    processed_df = preprocess_input(input_df, scaler, expected_columns)
    predictions = model.predict(processed_df)
    probabilities = model.predict_proba(processed_df)[:, 1]
    return predictions, probabilities

if __name__ == "__main__":
    model, scaler = load_artifacts(BEST_RUN_ID, MODEL_ARTIFACT_PATH, SCALER_PATH)
    try:
        df_processed_cols = pd.read_csv(PROCESSED_DATA_PATH, nrows=0).columns.tolist()
        expected_model_columns = [col for col in df_processed_cols if col != 'isFraud']
    except FileNotFoundError:
        print(f"Erro: Arquivo {PROCESSED_DATA_PATH} não encontrado. Não foi possível obter as colunas esperadas.")
        expected_model_columns = []
    if model and scaler and expected_model_columns:
        new_transaction = {
            'step': 1,
            'type': 'PAYMENT',
            'amount': 9839.64,
            'oldbalanceOrg': 170136.0,
            'newbalanceOrig': 160296.36,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'isFlaggedFraud': 0
        }
        prediction, probability = predict_fraud(new_transaction, model, scaler, expected_model_columns)
        if prediction is not None:
            print(f"\n--- Previsão para Nova Transação ---")
            print(f"Previsão (0=Não Fraude, 1=Fraude): {prediction[0]}")
            print(f"Probabilidade de ser Fraude: {probability[0]:.4f}")
        new_transactions_df = pd.DataFrame([
            {'step': 2, 'type': 'TRANSFER', 'amount': 181.00, 'oldbalanceOrg': 181.0, 'newbalanceOrig': 0.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0, 'isFlaggedFraud': 0},
            {'step': 3, 'type': 'CASH_OUT', 'amount': 229133.94, 'oldbalanceOrg': 15325.0, 'newbalanceOrig': 0.0, 'oldbalanceDest': 5083.0, 'newbalanceDest': 51513.44, 'isFlaggedFraud': 0}
        ])
        processed_df_batch = preprocess_input(new_transactions_df, scaler, expected_model_columns)
        predictions_batch, probabilities_batch = predict_fraud(new_transactions_df, model, scaler, expected_model_columns)
        if predictions_batch is not None:
            print(f"\n--- Previsão para Lote de Transações ---")
            print(f"Previsões: {predictions_batch}")
            print(f"Probabilidades: {probabilities_batch}")
        explainer = shap.TreeExplainer(model)
        processed_array_batch = processed_df_batch.values
        feature_names = processed_df_batch.columns.tolist()
        shap_values_batch = explainer.shap_values(processed_array_batch)
        if isinstance(shap_values_batch, list) and len(shap_values_batch) == 2:
            shap_values_class1 = shap_values_batch[1]
            expected_value_class1 = explainer.expected_value[1]
        elif isinstance(shap_values_batch, np.ndarray) and shap_values_batch.ndim == 3:
            shap_values_class1 = shap_values_batch[:, :, 1]
            expected_value_class1 = explainer.expected_value[1]
        else:
            raise ValueError(f"Formato inesperado para shap_values: {type(shap_values_batch)}")
        shap_values_dim = shap_values_class1.shape[1]
        if processed_array_batch.shape[1] == shap_values_dim:
            plt.figure(figsize=(20, 5))
            feature_values_1 = [f'{v:.2f}' if isinstance(v, float) else v for v in processed_array_batch[0, :]]
            abbr_feature_names = [name if len(name) <= 12 else name[:10] + '…' for name in feature_names]
            shap.force_plot(
                expected_value_class1,
                shap_values_class1[0, :],
                feature_values_1,
                feature_names=abbr_feature_names,
                matplotlib=True,
                show=False
            )
            plt.title("Explicação SHAP - Transação 1 (TRANSFER - Fraude)")
            for text in plt.gca().texts:
                try:
                    val = float(text.get_text())
                    text.set_text(f'{val:.2f}')
                except Exception:
                    pass
                text.set_fontsize(8)
            plt.tight_layout()
            plt.savefig("shap_force_plot_transacao1.png")
            plt.close()
            plt.figure(figsize=(20, 5))
            feature_values_2 = [f'{v:.2f}' if isinstance(v, float) else v for v in processed_array_batch[1, :]]
            shap.force_plot(
                expected_value_class1,
                shap_values_class1[1, :],
                feature_values_2,
                feature_names=abbr_feature_names,
                matplotlib=True,
                show=False
            )
            plt.title("Explicação SHAP - Transação 2 (CASH_OUT - Não Fraude)")
            for text in plt.gca().texts:
                try:
                    val = float(text.get_text())
                    text.set_text(f'{val:.2f}')
                except Exception:
                    pass
                text.set_fontsize(8)
            plt.tight_layout()
            plt.savefig("shap_force_plot_transacao2.png")
            plt.close()
            plt.figure()
            shap.summary_plot(shap_values_class1, processed_array_batch, feature_names=feature_names, show=False)
            plt.title("Importância Global das Features (SHAP)")
            plt.tight_layout()
            plt.savefig("shap_summary_plot.png")
            plt.close()
