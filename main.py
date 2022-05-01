from typing import Optional
import pandas as pd
from fastapi import FastAPI
from joblib import load
import DataModel
import PredictionModel


app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(dataModel: DataModel.DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    df = df.drop("Life expectancy", axis=1)
    model = load("modelo.joblib")
    result = model.predict(df)
    print(df)
    print(result)
    return result.tolist()


@app.post("/getrsquared")
def get_r2(dataModel: DataModel.DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    x = df.drop("Life expectancy", axis=1)
    y = df["Life expectancy"]
    model = load("modelo.joblib")
    result = model.score(x, y)
    print(x)
    print(y)
    print(result)
    return result

