from bin_db import BinDB
from datetime import datetime
from dotenv import load_dotenv
import os
import hashlib
import time

load_dotenv()

def calculate_file_sha256(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

# Start timing
start_time = time.time()

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

latest_file = 'simple_text.txt'
# latest_file = 'random_text.txt'
# latest_file = 'random_lorem_text.txt'
# latest_file = 'random_binary.bin'
binary_file_path = os.path.join('data', latest_file)

# Calculate file hash
file_sha256 = calculate_file_sha256(binary_file_path)
print(f"File SHA-256: {file_sha256}")
print(f"Indexing file: {binary_file_path}")

# db.index_entire_file(
#     file_path=binary_file_path,
#     file_sha256=file_sha256,
#     min_size=8,
#     max_size=8,
#     batch_size=100000,
#     # dry_run=True
# )

# Index the generated random binary file
# db.index_binary_file(
#     file_path=binary_file_path,
#     file_sha256=file_sha256,
#     min_size=8,
#     max_size=8,
#     batch_size=100000,
#     # dry_run=True
# ) 


# # Example Lookup
ngram = "his is ji"
results = db.search_by_ngram(ngram)

# Print results
for result in results:
    print(f"File: {result['file_path']}")
    print(f"SHA256: {result['file_sha256']}")
    print(f"Offset: {result['offset']}")
    print(f"Size: {result['size']}")
    print(f"Match Score: {result['score']}")
    print("---")
    
# Calculate and print total runtime
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal runtime: {total_time:.2f} seconds")

