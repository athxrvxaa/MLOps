from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.ecaluation import evaluate_model

@pipeline(enable_cache = True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    R2Score, RMSE = evaluate_model(model, X_test, y_test)