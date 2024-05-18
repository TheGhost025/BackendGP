from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from typing import List, Dict

cred = credentials.Certificate("ar-furniture-789f5-firebase-adminsdk-w4jtf-eb60bd3446.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = FastAPI()

@app.get('/')
def index():
    # Reference to the 'users' collection
    collection_ref = db.collection("users")

    # Fetch all documents in the collection
    docs = collection_ref.stream()

    # Convert documents to a list of dictionaries
    users: List[Dict] = []
    for doc in docs:
        user_data = doc.to_dict()
        user_data['id'] = doc.id  # Add the document ID to the data
        users.append(user_data)

    return users