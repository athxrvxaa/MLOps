from pipelines.training_pipeline import train_pipeline
from zenml.client import Client 


if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(r"/home/atharva/MLOps/mlops-mine/data/olist_customers_dataset.csv")
    print("Pipeline execution completed.")

# mlflow ui --backend-store-uri file:/home/atharva/.config/zenml/local_stores/77ef7d2d-fdc0-4bac-88c6-602765488386/mlruns