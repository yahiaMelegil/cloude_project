Live Application

🔗 Project URL (Ngrok):
https://subrhombic-maida-invariably.ngrok-free.dev/

Access the system using any modern browser — no login required.

🧠 Project Overview

This project implements a cloud-based distributed data processing service that enables users to:

Upload datasets in common formats

Compute descriptive statistics

Run machine learning jobs

Evaluate performance and scalability using Apache Spark

The system is deployed on Google Cloud Platform (GCP) with an interactive web interface built using Streamlit, and uses Apache Spark / PySpark for distributed processing.

🗂 Project Structure
📦 project-root
├── app.py
├── requirements.txt
├── README.md
├── utils/
│   ├── config.py
│   ├── file_handler.py
│   └── data_validator.py
├── spark_jobs/
│   ├── statistics.py
│   ├── ml_models.py
│   └── performance_test.py
├── data/
│   ├── uploads/
│   └── results/
└── .env (optional)
