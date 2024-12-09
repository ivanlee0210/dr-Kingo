## Download module for dr Kingo
## Class HAI5016 - by camphouse.me

# %%
from loguru import logger
from dotenv import load_dotenv
import os
from database import get_engine, get_session, skkuHtml
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import stealth_requests as requests
import time
import random
from tqdm import tqdm
from datetime import datetime
import hashlib

logger.remove()
logger.add("logs/downloader.log", rotation="10 MB", enqueue=True, backtrace=True, diagnose=True)
load_dotenv()

download_folder = "temp/skku"
base_url = "https://www.skku.edu/eng/"

engine = get_engine()
session = get_session(engine)

def get_url_df(table_name):
    try:
        url_df = pd.read_sql(table_name, engine, dtype=str)
        logger.info(f"{len(url_df)} Urls imported from SQL database")
        return url_df
    except Exception as e:
        logger.error(f"Failed to read from SQL database: {e}")
        raise


def download_urls(url_df):
    counts = {"insert": 0, "update": 0, "failed": 0}
    for index, row in tqdm(url_df.iterrows(), total=url_df.shape[0]):
        url = str(row['url'])

        tqdm.write(f"Processing {url}")

        # Check if the URL is valid
        if not url.startswith(base_url):
            logger.warning(f"Invalid URL: {url}")
            continue

        # Remove newline characters from the url
        logger.debug(f"Downloading {url}")
        url = url.strip()

        # Extract the path after base_url
        path = url.split(base_url)[-1]
        
        # Create the full path for the file
        file_path = os.path.join(download_folder, path)
        
        # Ensure the directory exists
        if file_path.endswith('/'):
            file_path = os.path.join(file_path, 'index.html')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Download the file
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)

        # Get the content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        cont_wrap_div = soup.find('div', class_='cont_wrap')

        # Make a hash of the content of the div
        cont_wrap_hash = hashlib.sha256(cont_wrap_div.text.encode('utf-8')).hexdigest()
        logger.debug(f"SHA-256 hash of the content: {cont_wrap_hash}")

        # Check if the record already exists
        logger.debug(f"Checking if {url} exists in the database")
        record = session.query(skkuHtml).filter(skkuHtml.url == url).first()

        if record:
            logger.debug(f"Record for URL {url} found in the database.")
            # Check if the hash matches
            if record.cont_wrap_hash == cont_wrap_hash:
                logger.debug(f"The hash for {url} matches. No further action required.")
                time.sleep(random.uniform(0.5, 2))
            else:
                logger.info(f"The hash for {url} does not match. Updating the record.")
                try:
                    # Update the existing record
                    record.html = response.content.decode('utf-8')
                    record.cont_wrap_hash = cont_wrap_hash
                    record.scraping_date = datetime.utcnow()
                    session.commit()
                    counts["update"] += 1
                    logger.info(f"Record for {url} updated successfully.")
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to update the record for {url}. Error: {str(e)}")
        else:
            logger.info(f"No record found for URL {url}. Inserting a new record.")
            # Insert the new record
            try:
                new_record = skkuHtml(
                    url=url,
                    html=response.content.decode('utf-8'),
                    cont_wrap_hash=cont_wrap_hash,
                    scraping_date=datetime.utcnow(),
                )
                session.add(new_record)
                session.commit()
                counts["insert"] += 1
                logger.info(f"New record for URL {url} inserted successfully.")
            except IntegrityError:
                session.rollback()
                counts["failed"] += 1
                logger.error(f"Failed to insert the record for URL {url} due to a conflict.")
            except Exception as e:
                session.rollback()
                counts["failed"] += 1
                logger.error(f"An unexpected error occurred while inserting {url}. Error: {str(e)}")
    return counts

def download_all():
    url_df = get_url_df("skku_urls")
    counts = download_urls(url_df)
    logger.info(f"Record counts: {counts}")

if __name__ == "__main__":
    logger.info("Downloader module run directly.")
    print("Starting download...")
    results = download_all()
    logger.info("Downloader module run completed.")
    print(f"Downloading complete: {counts}")
