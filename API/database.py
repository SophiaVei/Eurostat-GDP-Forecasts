from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "skillscapes")

class Database:
    client: MongoClient = None
    db = None

    def connect(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        print(f"Connected to MongoDB at {MONGO_URI}, DB: {DB_NAME}")

    def close(self):
        if self.client:
            self.client.close()

db = Database()

def get_database():
    return db.db
