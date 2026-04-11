"""
Data Ingestion Pipeline
Downloads the global petrol prices dataset from Kaggle and moves it
from the local cache to the project's 'data' directory.
"""

import kagglehub
import shutil
import logging
from pathlib import Path

# Configure professional logging (Senior level practice)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def ingest_kaggle_dataset() -> None:
    """
    Downloads the specified Kaggle dataset and copies the files 
    to the local project data directory for reproducibility.
    """
    dataset_id = "zkskhurram/global-petrol-prices-impact-of-2026-us-iran-war"
    
    # Define paths using pathlib (Cross-platform compatibility)
    project_root = Path(__file__).parent
    target_dir = project_root / "data"
    
    # Create the data directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Initiating download for dataset: '{dataset_id}'")
    
    try:
        # Download to Kaggle's default cache
        cache_path = kagglehub.dataset_download(dataset_id)
        cache_dir = Path(cache_path)
        
        logging.info(f"Successfully downloaded to cache: {cache_dir}")
        logging.info(f"Transferring files to project directory: {target_dir}")
        
        # Iterate through the downloaded files and copy them to our data folder
        file_count = 0
        for file_path in cache_dir.iterdir():
            if file_path.is_file():
                target_file = target_dir / file_path.name
                shutil.copy2(file_path, target_file) # copy2 preserves metadata
                logging.info(f"Copied: {file_path.name}")
                file_count += 1
                
        logging.info(f"✅ Data ingestion complete. {file_count} files securely moved to /data.")
        
    except Exception as e:
        logging.error(f"❌ Data ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    ingest_kaggle_dataset()