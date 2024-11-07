# MLOps (DevOps for ML)

### Acknowledgements
---
I would like to extend my sincere thanks to [freeCodeCamp](https://www.freecodecamp.org/) and [Ayush Singh](https://www.youtube.com/@AyushSinghSh) for their invaluable content and guidance in helping me build this project. This project wouldn't have been possible without their educational resources.

<br>
<br>

### Introduction to MLOps
---
**MLOps**, short for [Machine Learning Operations](https://aws.amazon.com/what-is/mlops/#:~:text=Machine%20learning%20operations%20(MLOps)%20are,deliver%20value%20to%20your%20customers.), brings the practices of DevOps to the world of machine learning, helping teams build, deploy, and maintain ML models more effectively. By automating repetitive tasks like testing, deployment, and monitoring, MLOps allows data scientists and engineers to focus on improving model accuracy and performance. It makes scaling models easier, keeps them reliable over time, and ensures that they stay relevant by catching issues early—like when a model's accuracy drops or when it starts to drift from real-world data. MLOps bridges the gap between development and production, making sure that the hard work of building models actually makes it into the hands of users smoothly and effectively.

<br>
<br>

### About the Project
---
This MLOps project focuses on **Customer Satisfaction** *for purchased product(s)*. While the problem statement is simple, the primary purpose is to explore how MLOps principles—essentially applying DevOps practices to machine learning—enhance any ML project. **MLOps**, short for Machine Learning Operations, guides best practices in automating and monitoring ML models in production. This course will walk you through an end-to-end MLOps project, covering data ingestion through deployment, using cutting-edge tools like [ZenML](https://www.zenml.io/), [MLflow](https://mlflow.org/), and other essential MLOps libraries.

<br>
<br>

### Structure of the Project Directories and Files
---
```plaintext 
MLOps-DevOps-for-ML/
│ 
├──  data/
│	└── dataset.csv
│ 
├── pipelines/ 
│ 	└── deployment_pipeline.py
│	└── training_pipeline.py
│	└── utils.py
│
├── src/ 
│ 	└── data_cleaning.py
│ 	└── evaluation.py
│ 	└── model_dev.py
│
├── steps/ 
│ 	└── clean_data.py
│ 	└── config.py
│ 	└── evaluation.py
│ 	└── ingest_data.py
│ 	└── model_train.py
│
├──  requirements.txt
├──  run_deployment.py
├──  run_pipeline.py
└──  streamlit_app.py
```

<br>
<br>

### Running Locally
---
🌟 Please got step-wise, otherwise in no time your brain will 🤯💥
<br>

 - Create a `conda` environment (with `Python = 3.10` version installed)  : `conda create --n customer-satisfaction python=3.10`
 - **(Alternative to the above)** ▶️ If you want to keep the `conda` environment local to the project directory, you may go with : `conda create --prefix ./customer-satisfaction python=3.10`
 - Activate the `conda` environment : `conda activate ./customer-satisfaction`
  (NOTE : You are free to use any name for your `conda` environment)
 - Install all the required libraries (list mentioned in `requirements.txt`) on Terminal / CLI / Anaconda Prompt : `pip install -r requirements.txt`
 - IMPORTANT NOTE 🔴 : `data` directory is not present as the dataset is pretty huge to upload. Thus, create a `data` directory in the `root` directory. Inside you will have to store the dataset ([LINK]())
