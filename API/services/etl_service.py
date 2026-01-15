import pandas as pd
import sys
from pathlib import Path
from ..database import get_database

# Import existing logic
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processor import preprocess_to_wide_file
from download_data import (
    download_economy_default, 
    download_labour_default, 
    download_tourism_default, 
    download_greek_tourism_default
)

def update_datasets():
    """
    Downloads raw data, processes to wide format, and saves to MongoDB.
    """
    db = get_database()
    datasets_collection = db["datasets"]
    
    # 1. Trigger Downloads
    print("Downloading latest data...")
    download_economy_default()
    download_labour_default()
    download_tourism_default()
    download_greek_tourism_default()
    
    # 2. Process and Save
    domains = ['economy', 'labour', 'tourism', 'greek_tourism']
    repo_dir = SRC_DIR.parent
    data_dir = repo_dir / "data"
    
    for domain in domains:
        raw_file = data_dir / f"{domain}_nuts2_all_columns.csv"
        wide_file = data_dir / f"{domain}_nuts2_wide.csv"
        
        if raw_file.exists():
            print(f"Processing {domain}...")
            # Preprocess
            wide_df = preprocess_to_wide_file(str(raw_file), str(wide_file))
            
            # Save to MongoDB (Split by Domain Collection)
            # Collection Name = domain (e.g. "economy")
            collection = db[domain]
            
            # Delete existing "history" for this domain to avoid duplication
            collection.delete_many({"type": "history"})
            
            records = wide_df.to_dict(orient='records')
            mongo_docs = []
            for r in records:
                geo = r.pop('geo', 'unknown')
                mongo_docs.append({
                    "geo": geo,
                    "type": "history",  # Distinguish from forecasts
                    "data": r,
                    "updated_at": pd.Timestamp.now()
                })
            
            if mongo_docs:
                collection.insert_many(mongo_docs)
                print(f"Saved {len(mongo_docs)} history records to collection '{domain}'.")
        else:
            print(f"Warning: Raw file for {domain} not found.")

    return {"status": "success", "message": "Datasets updated in MongoDB"}
