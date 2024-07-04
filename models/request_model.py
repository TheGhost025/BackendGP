from pydantic import BaseModel
from typing import Union


class PredictRequest(BaseModel):
    features: list


class PredictMultipleRequest(BaseModel):
    products: list[list[Union[int, float]]]
