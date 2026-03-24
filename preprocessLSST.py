import pandas as pd
from sqlalchemy import create_engine
import os
import sys

def preprocess_db(db_path):
    if not os.path.exists(db_path):
        print(f"Error: File not found: {db_path}")
        return
    
    print(f"Loading data from {db_path}...")
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Read the database just like in appHealpy
        query = "SELECT fieldRa, fieldDec, observationStartMJD, flush_by_MJD, visitExposureTime, band FROM observations where scheduler_note like 'pair%'"
        df = pd.read_sql(query, engine)
            
        # Determine output CSV filename
        base_name = os.path.splitext(db_path)[0]
        csv_path = f"{base_name}.csv"
        
        print(f"Saving {len(df)} rows to {csv_path}...")
        df.to_csv(csv_path, index=False)
        print("Done!")
        
    except Exception as e:
        print(f"Error processing database: {e}")

if __name__ == "__main__":
    db_file = './baseline_v5.1.1_10yrs.db'
    if len(sys.argv) > 1:
        db_file = sys.argv[1]
    
    preprocess_db(db_file)
