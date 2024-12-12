# %% [markdown]
# ## Load the libraries

# %%
from loguru import logger
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from database import get_engine, get_session, skkuHtml, skkuMd
from tqdm import tqdm
import json
import pandas as pd
import hashlib
import requests

# %% [markdown]
# ## Create logger, load environment variables and database connection
# 

# %%
## Load environment variables from .env file
load_dotenv()

## Set up logger
logger.remove()
logger.add("logs/markdown_parser.log", rotation="10 MB")

engine = get_engine()
session = get_session(engine)

# %% [markdown]
# ## Load the data from the skku_html table

# %%
# Query all records from the skkuHtml table and get the url and html_wrap_hash fields
logger.debug("Loading url and html_wrap_hash fields from all records in the skkuHtml table.")
try:
    records = session.query(skkuHtml.url, skkuHtml.cont_wrap_hash).all()
    logger.info(f"Loaded {len(records)} records from the skku_html table.")
    
    # Convert the records to a dataframe
    html_records = pd.DataFrame(records, columns=['url', 'cont_wrap_hash'])
except Exception as e:
    logger.error(f"Failed to load records from skkuHtml. Error: {str(e)}")

# %% [markdown]
# ## Load the data from the skku_md table

# %%
# Due the same for all records in the skku_md table, and get the url and html_wrap_hash fields
logger.debug("Loading url and html_wrap_hash fields from all records in the skku_md table.")
try:
    records = session.query(skkuMd.url, skkuMd.html_wrap_hash).all()
    logger.info(f"Loaded {len(records)} records from the skkuMd table.")
    
    # Convert the records to a dataframe
    md_records = pd.DataFrame(records, columns=['url', 'html_wrap_hash'])
except Exception as e:
    logger.error(f"Failed to load records from skkuHtml. Error: {str(e)}")

# %% [markdown]
# ## Compare those two tables and save the differences

# %%
# Create a dataframe to_parse that contains all the records from the skkuHtml table that do not match any records in the skkuMd table
logger.debug("Creating a dataframe to_parse that contains all the records from the skkuHtml table that do not match any records in the skkuMd table.")
try:
    to_parse = html_records[~html_records['cont_wrap_hash'].isin(md_records['html_wrap_hash'])]
    logger.info(f"Created a dataframe to_parse with {len(to_parse)} records.")
except Exception as e:
    logger.error(f"Failed to create a dataframe to_parse. Error: {str(e)}")

logger.info(f"{len(to_parse)} urls to parse to Markdown format.")

# %% [markdown]
# ## Create Azure OpenAI instance and test inference

# %%
from openai import AzureOpenAI, OpenAIError

aoi_api_key = os.getenv("AZURE_OPENAI_API_KEY")
aoi_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
aoi_gen_model = os.getenv("AZURE_GENERATION_MODEL")
aoi_version = os.getenv("AZURE_GENERATION_MODEL_VERSION")

logger.info(f"Creating an instance of the AzureOpenAI class with the following parameters: endpoint={aoi_endpoint}, api_version={aoi_version}")

client = AzureOpenAI(azure_endpoint=aoi_endpoint,api_key=aoi_api_key, api_version=aoi_version)

# %%
## Test the completion API

completion = client.chat.completions.create(
    model=aoi_gen_model, # This must match the custom deployment name you chose for your model.
    messages=[
        {"role": "user", "content": "How ya feeling today?"},
    ],
)

print(completion.choices[0].message.content)

# %% [markdown]
# ## Function to dynamically generate a prompt

# %%
def create_prompt(prompt_file, html_content):
	logger.info(f"Creating a prompt based on {prompt_file} and the HTML content.")
	# Load the prompt file
	with open(prompt_file, 'r') as f:
		prompt = f.read()
		# Add the HTML content to the prompt
		prompt += f"\n\n{html_content}\n"
	return prompt

# Function to make the Azure OpenAI API call and generate the markdown output
def generate_markdown_from_html(prompt_file, html_content, timeout=60):
    prompt = create_prompt(prompt_file, html_content)
    
    try:
        completion = client.chat.completions.create(
            model=aoi_gen_model, 
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout  # Set the timeout here
        )
        logger.debug(completion.usage)
        # Return the generated markdown content
        return completion
    except requests.exceptions.Timeout:
        logger.error("The request timed out.")
        return None
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return None

# %% [markdown]
# ## Function for title and html content extraction

# %%
def process_title(title):
    # Split the title by '|'
    segments = title.split('|')
    
    # Remove the first segment (Sungkyunkwan University)
    segments = segments[1:]
    
    # Keep only the last three segments
    if len(segments) > 3:
        segments = segments[-3:]
    
    # Join the segments back with '|'
    processed_title = ' | '.join(segment.strip() for segment in segments)
    
    return processed_title

def extract_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = process_title(soup.title.string)
    cont_wrap_div = soup.find('div', class_='cont_wrap')
    return {'title': title, 'cont_wrap': cont_wrap_div}

# %%
# # Get 10 random rows from the DataFrame
# to_parse = to_parse.sample(n=10, random_state=1)

# %% [markdown]
# ## Create the Markdown content from HTML files and save into the database

# %%
for index, row in tqdm(to_parse.iterrows(), total=to_parse.shape[0], desc="Processing records"):
    url = row['url']
    cont_wrap_hash = row['cont_wrap_hash']

    ## From the skkuHtml get the html content for the given url
    logger.debug(f"Loading HTML content for {url}.")
    try:
        html_content = session.query(skkuHtml.html).filter(skkuHtml.url == url).first()
        html_content_str = html_content[0]  # Get the HTML content as a string
        logger.info(f"Loaded HTML content for {url}.")
    except Exception as e:
        logger.error(f"Failed to load HTML content for {url}. Error: {str(e)}")

    ## Extract the content from the HTML
    logger.debug("Extracting content from the HTML.")
    page_title = extract_content(html_content_str)['title']

    logger.debug(f"Page title: {page_title}")

    # Inject the title as <h1> into the HTML content
    cont_wrap_div = extract_content(html_content_str)['cont_wrap']
    cont_wrap_div = f"<h1>{page_title}</h1>\n{cont_wrap_div}"

    # Generate Markdown from the HTML content
    prompt_path = "prompts/1.txt"
    try:
        completion = generate_markdown_from_html(prompt_path, cont_wrap_div)
        usage = json.dumps(completion.usage.model_dump())  # Convert the usage dictionary to a JSON string
        total_tokens = completion.usage.total_tokens
        model = completion.model
        created = completion.created
        markdown = completion.choices[0].message.content
        logger.info(f"Generated markdown for URL: {url}")
    except Exception as e:
        logger.error(f"Failed to generate markdown for URL: {url}. Error: {str(e)}")
        continue

    # Calculate the hash of the markdown content
    md_wrap_hash = hashlib.md5(markdown.encode()).hexdigest()

    try:
        md_record = skkuMd(url=row['url'], html_wrap_hash=row['cont_wrap_hash'], md_wrap_hash=md_wrap_hash, markdown=markdown, usage=usage, model=model, created=created, total_tokens=total_tokens)
        session.merge(md_record)
        session.commit()
        logger.info(f"Markdown content saved to skkuMd table for URL: {url}")
    except IntegrityError as e:
        session.rollback()
        logger.error(f"Failed to save markdown content to skkuMd table for URL: {url}. Error: {str(e)}")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Failed to save markdown content to skkuMd table for URL: {url}. Error: {str(e)}")


# %%



