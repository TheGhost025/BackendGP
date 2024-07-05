from sklearn.neighbors import NearestNeighbors
import traceback
from fastapi import HTTPException
from methods.fetch_methods import fetch_products_from_firebase
from firebase_admin import db
import pandas as pd
import numpy as np
from methods.encoder_method import initialize_encoders

# Instantiate the KNN NearestNeighbour model
knn = NearestNeighbors(n_neighbors=7)


def train():
    try:
        X_train = fetch_products_from_firebase()
        # print(X_train)
        knn.fit(X_train)
        return {"status": "training successful"}
    except Exception as e:
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


def predict_by_id(product_id: str):
    try:
        # Initialize encoders
        encoders = initialize_encoders()

        # Get the product from the database
        ref = db.reference(f'products/{product_id}')
        product = ref.get()

        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        # Extract the features from the product
        keys = ['category', 'material', 'price', 'sizeX', 'sizeY', 'sizeZ', 'weight']
        product_features = []
        for key in keys:
            value = product.get(key, 'N/A')
            if key in ['category', 'material']:
                if value != 'N/A':
                    value = encoders[key].transform([value])[0]
            else:
                try:
                    value = pd.to_numeric(value)
                except ValueError:
                    value = 0  # default value
            product_features.append(value)

        # Transform the features into a numpy array and reshape it
        X_test = np.array(product_features).reshape(1, -1)

        # Get the indices of the nearest neighbors and their distances
        distances, indices = knn.kneighbors(X_test)

        # Get all products from the database
        ref = db.reference('products')
        products = ref.get()
        product_keys = list(products.keys())

        # Prepare the response
        response = []
        for i in range(len(indices[0])):
            index = indices[0][i]
            distance = distances[0][i]
            neighbor_product_id = product_keys[index]
            # Skip the neighbor if its id is the same as the product_id
            if neighbor_product_id == product_id or distance > 1500:
                continue
            neighbor_product = products[neighbor_product_id]
            response.append({
                "product": neighbor_product,
                "distance": distance
            })

        return response
    except Exception as e:
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))