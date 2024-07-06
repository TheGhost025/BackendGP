from datetime import datetime

import uvicorn
from fastapi import FastAPI

from methods.fetch_methods import *
from methods.ml_methods import *
from models.request_model import *
from models.user_model import *

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


@app.post("/users/add-edit/")
def create_user(user: NewUser):
    try:
        # Access the Firestore collection
        user_ref = dbs.collection("users").document(user.user_id)
        user_ref.get()
        if user_ref.get().exists:
            new_or_edit = 1
            user_ref.set(
                user.dict(by_alias=True,
                          exclude={"cart", "comment_and_reviews", "favorite", "history_purchased", "user_id"}))
        else:
            new_or_edit = 0
            user_ref.set(
                user.dict(by_alias=True,
                          exclude={"cart", "comment_and_reviews", "favorite", "history_purchased", "user_id"}))

        # Create collections by default
        user_ref.collection('cart')
        user_ref.collection('comment and reviews')
        user_ref.collection('favorite')
        user_ref.collection('history_purchased')

        if user.cart:
            cart_ref = user_ref.collection('cart')
            for item in user.cart:
                item['date'] = datetime.now()
                cart_ref.document(item['product_id']).set(item)
        if user.comment_and_reviews:
            comments_and_reviews_ref = user_ref.collection('comment and reviews')
            for item in user.comment_and_reviews:
                item['date'] = datetime.now()
                comments_and_reviews_ref.document(item['product_id']).set(item)
        if user.favorite:
            favorite_ref = user_ref.collection('favorite')
            for item in user.favorite:
                item['date'] = datetime.now()
                favorite_ref.document(item['product_id']).set(item)
        if user.history_purchased:
            history_purchased_ref = user_ref.collection('history_purchased')
            for item in user.history_purchased:
                item['date'] = datetime.now()
                history_purchased_ref.document(item['product_id']).set(item)

        if new_or_edit:
            return {"message": "User edited successfully"}
        else:
            return {"message": "User created successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------- Machine Learning --------------------


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


@app.post("/predict_multiple/")
async def predict_multiple(request: PredictMultipleRequest):
    try:
        # Prepare the response
        response = []

        # For each product in the request
        for product_features in request.products:
            # Transform the features into a numpy array and reshape it
            X_test = np.array(product_features).reshape(1, -1)

            # Get the indices of the nearest neighbors and their distances
            distances, indices = knn.kneighbors(X_test)

            # Get the products from the database
            ref = db.reference('products')
            products = ref.get()
            product_keys = list(products.keys())

            # For each neighbor
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


@app.get("/predict_by_id/{product_id}")
async def make_prediction_by_id(product_id: str):
    # Train the Model
    train()

    return predict_by_id(product_id)


@app.get("/user_products")
async def get_user_products(user: User):
    return fetch_user_product_ids(user.user_id)


@app.get("/user_products_neighbours/{user_id}")
async def get_user_products_neighbours(user_id: str):
    # Train the Model
    train()

    # Fetch the product IDs associated with the user
    user_product_ids_dict = fetch_user_product_ids(user_id)

    # Create a list of all product IDs
    all_product_ids = list(user_product_ids_dict.keys())

    # Create a dictionary to store the neighbours
    neighbours_dict = {}

    # For each product ID
    for product_id, count in user_product_ids_dict.items():
        try:
            # Get the nearest neighbors for the product
            neighbours = predict_by_id(product_id)

            # Remove any neighbours that are in the list of all product IDs
            neighbours = [neighbour for neighbour in neighbours if neighbour['product']['id'] not in all_product_ids]

            # Add the neighbours to the dictionary
            for neighbour in neighbours:
                neighbours_dict[neighbour['product']['id']] = neighbour

        except HTTPException as e:
            print(f"Product {product_id} not found. Skipping...")

    # Convert the dictionary of neighbours back to a list
    neighbours_list = list(neighbours_dict.values())

    # Add the neighbours to the response
    response = {
        "neighbours": neighbours_list
    }

    return response
