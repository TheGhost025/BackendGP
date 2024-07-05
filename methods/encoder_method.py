from sklearn.preprocessing import LabelEncoder
from firebase_admin import db


def initialize_encoders():
    ref = db.reference('products')
    products = ref.get()
    categories = {key: set() for key in ['category', 'material']}
    if products:
        for key, value in products.items():
            for key1 in ['category', 'material']:
                value1 = value.get(key1, 'N/A')
                if value1 != 'N/A':
                    categories[key1].add(value1)
    encoders = {key: LabelEncoder().fit(list(val)) for key, val in categories.items()}
    return encoders
