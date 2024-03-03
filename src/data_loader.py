import pandas as pd
from sqlalchemy import create_engine
from config import Config

def load_data():
    engine = create_engine(Config.DATABASE_URL)
    query = "SELECT * FROM lung_cancer"
    data = pd.read_sql(query, engine)

    print("Data loaded successfully:")

    print(data.head())

    return data