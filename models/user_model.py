from pydantic import BaseModel, EmailStr
from typing import List, Dict


class NewUser(BaseModel):
    user_id: str
    firstName: str
    lastName: str
    address: str
    birthofdate: str
    gender: str
    optionaladdress: str
    usertype: str
    email: EmailStr
    cart: List[Dict] = []
    comment_and_reviews: List[Dict] = []
    favourite: List[Dict] = []
    history_purchased: List[Dict] = []


class User(BaseModel):
    user_id: str