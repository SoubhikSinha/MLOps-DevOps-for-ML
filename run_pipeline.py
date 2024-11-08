'''
Running the Pipeline
'''

# Importing the necessary libraries
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

# Running the pipeline
if __name__ == "__main__":
    # Running the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    train_pipeline(data_path = "D:\GitHub_Repos\MLOps-DevOps-for-ML\data\olist_customers_dataset.csv")