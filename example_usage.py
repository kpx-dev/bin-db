from bin_db import BinDB
from datetime import datetime
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

def calculate_file_sha256(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

# Get configuration from environment variables
endpoint = os.getenv('OPENSEARCH_ENDPOINT')
region = os.getenv('AWS_REGION')
index_name = os.getenv('INDEX_NAME')

if not all([endpoint, region, index_name]):
    raise ValueError("Missing required environment variables. Please check your .env file")

db = BinDB(
    collection_endpoint=endpoint,
    region=region,
    index_name=index_name
)
# db.delete_index()
# db.create_index()

latest_file = 'random_text.txt'
binary_file_path = os.path.join('data', latest_file)

# Calculate file hash
file_sha256 = calculate_file_sha256(binary_file_path)
print(f"File SHA-256: {file_sha256}")
print(f"Indexing file: {binary_file_path}")

# Index the generated random binary file
# db.index_binary_file(
#     file_path=binary_file_path,
#     file_sha256=file_sha256,
#     min_size=4,
#     max_size=8,
#     batch_size=1000
# ) 

# Example usage
ngram = "2f555762"
results = db.search_by_ngram(ngram)

# Print results
for result in results:
    print(f"File: {result['file_path']}")
    print(f"SHA256: {result['file_sha256']}")
    print(f"Offset: {result['offset']}")
    print(f"Size: {result['size']}")
    print(f"Match Score: {result['score']}")
    print("---")