import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, '../data/lung_cancer.db')}"
    MODEL_SAVE_PATH = "decision_tree_model.pkl"