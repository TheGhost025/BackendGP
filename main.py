from fastapi import FastAPI, HTTPException
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import firestore
from typing import List, Dict
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import traceback
from sklearn.preprocessing import LabelEncoder


# Initialize Firebase Admin
cred = credentials.Certificate("ar-furniture-789f5-firebase-adminsdk-w4jtf-e20c880149.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ar-furniture-789f5-default-rtdb.firebaseio.com/'
})

dbs = firestore.client()

app = FastAPI()


@app.get('/')
def index():
    # Reference to the 'users' collection
    collection_ref = dbs.collection("users")

    # Fetch all documents in the collection
    docs = collection_ref.stream()

    # Convert documents to a list of dictionaries
    users: List[Dict] = []
    for doc in docs:
        user_data = doc.to_dict()
        user_data['id'] = doc.id  # Add the document ID to the data
        users.append(user_data)

    return users


# Instantiate the KNN NearestNeighbour model with k=3
knn = NearestNeighbors(n_neighbors=3)


def fetch_data_from_firebase():
    ref = db.reference('products')
    products = ref.get()
    print("products", products)
    X_train = []
    keys = [
        'category',
        # 'description',
        'material',
        'price',
        # 'productName',
        'sizeX',
        'sizeY',
        'sizeZ',
        'weight'
    ]
    categories = {key: set() for key in ['category', 'material']}
    if products:
        for key, value in products.items():
            for key1 in ['category', 'material']:
                value1 = value.get(key1, 'N/A')
                if value1 != 'N/A':
                    categories[key1].add(value1)
    encoders = {key: LabelEncoder().fit(list(val)) for key, val in categories.items()}
    if products:
        for key, value in products.items():
            product_features = []
            for key1 in keys:
                value1 = value.get(key1, 'N/A')
                if key1 in ['category', 'material']:
                    if value1 != 'N/A':
                        value1 = encoders[key1].transform([value1])[0]
                else:
                    try:
                        value1 = pd.to_numeric(value1)
                    except ValueError:
                        value1 = 0  # default value
                product_features.append(value1)
            X_train.append(product_features)
    print("train", X_train)
    return np.array(X_train)


@app.post("/train/")
async def train():
    try:
        X_train = fetch_data_from_firebase()
        print(X_train)
        knn.fit(X_train)
        return {"status": "training successful"}
    except Exception as e:
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


class PredictRequest(BaseModel):
    features: list


@app.post("/predict/")
async def predict(request: PredictRequest):
    try:
        # Transform the features into a numpy array and reshape it
        X_test = np.array(request.features).reshape(1, -1)

        # Get the indices of the nearest neighbors and their distances
        distances, indices = knn.kneighbors(X_test)

        # Get the products from the database
        ref = db.reference('products')
        products = ref.get()
        product_keys = list(products.keys())

        # Prepare the response
        response = []
        for i in range(len(indices[0])):
            index = indices[0][i]
            distance = distances[0][i]
            product = products[product_keys[index]]
            response.append({
                "product": product,
                "distance": distance
            })

        return response
    except Exception as e:
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


