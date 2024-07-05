from firebase_admin import db
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter
from firebase_admin import firestore
from firebase_admin import credentials
import firebase_admin


def fetch_products_from_firebase():
    ref = db.reference('products')
    products = ref.get()
    # print("products", products)
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
    # print("train", X_train)
    return np.array(X_train)


# Initialize Firebase Admin
cred = credentials.Certificate("ar-furniture-789f5-firebase-adminsdk-w4jtf-53393ec8c3.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ar-furniture-789f5-default-rtdb.firebaseio.com/'
})

dbs = firestore.client()


def fetch_user_product_ids(user_id: str):
    collections = ["cart", "comment and reviews", "favourite", "history_purchased"]
    product_ids = []

    for collection in collections:
        # Access the Firestore collection
        doc_ref = dbs.collection("users").document(user_id).collection(collection)
        docs = doc_ref.stream()

        for doc in docs:
            product_ids.append(doc.id)

    # Convert the list of product_ids into a dictionary
    product_ids_dict = dict(Counter(product_ids))

    # Sort the dictionary by count in descending order
    product_ids_dict = dict(sorted(product_ids_dict.items(), key=lambda item: item[1], reverse=True))

    return product_ids_dict
