import json
import struct
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from requests_aws4auth import AWS4Auth

# 1. Initialize AWS Boto3 client for OpenSearch Serverless and credentials
region = "us-east-1"  # Replace with your AWS region
domain_endpoint = "88e8srnxk6448i46lkp8.us-east-1.aoss.amazonaws.com"  # Replace with your OpenSearch Serverless endpoint

# Use boto3 Session to automatically load AWS credentials
session = boto3.Session()
credentials = session.get_credentials()

# Get credentials
aws_access_key = credentials.access_key
aws_secret_key = credentials.secret_key
aws_session_token = credentials.token

# Create AWS4Auth for OpenSearch authentication
aws_auth = AWS4Auth(aws_access_key, aws_secret_key, region, 'aoss', session_token=aws_session_token)

# OpenSearch Serverless endpoint URL
opensearch_url = f"https://{domain_endpoint}"

# Create OpenSearch client with AWS authentication for OpenSearch Serverless (AOSS)
opensearch_client = OpenSearch(
    hosts=[{'host': domain_endpoint, 'port': 443}],
    http_auth=aws_auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# 2. Create the index with custom ngram analyzer
def create_index():
    # Define the ngram analyzer with a step size of 1 for 4-byte n-grams
    index_definition = {
        "settings": {
            "analysis": {
                "tokenizer": {
                    "ngram_tokenizer": {
                        "type": "ngram",
                        "min_gram": 4,
                        "max_gram": 4,
                        # "token_chars": ["letter", "digit"]
                    }
                },
                "analyzer": {
                    "ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "ngram_tokenizer"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "ngram": {
                    "type": "text",
                    "analyzer": "ngram_analyzer"
                },
                "packed_ngram": {
                    "type": "long"  # We will store the n-grams as long integers
                }
            }
        }
    }

    # Create index in OpenSearch Serverless (AOSS) with the name "packed_integers_index"
    response = opensearch_client.indices.create(index='packed_integers_index', body=index_definition, ignore=400)
    
    if response.get('acknowledged', False):
        print("Index 'packed_integers_index' created successfully!")
    else:
        print(f"Error creating index: {response}")

# 3. Ingest the binary file's 4-byte ngrams as packed integers into OpenSearch
def ingest_ngrams(file_name):
    # Generate 4-byte n-grams from the binary file
    def extract_ngrams(file_name, ngram_size=4, step_size=1):
        ngrams = []
        with open(file_name, "rb") as f:
            file_data = f.read()
            for i in range(0, len(file_data) - ngram_size + 1, step_size):
                ngram = file_data[i:i + ngram_size]
                ngrams.append(ngram)
        return ngrams

    # Convert 4-byte n-grams to packed integers
    def convert_ngrams_to_integers(ngrams):
        packed_integers = []
        for ngram in ngrams:
            packed_int = struct.unpack(">Q", ngram.ljust(8, b'\0'))[0]  # Pad with 0s to fit 64-bit
            packed_integers.append(packed_int)
        return packed_integers

    # Extract and convert the n-grams
    ngrams = extract_ngrams(file_name)
    packed_ngrams = convert_ngrams_to_integers(ngrams)

    # Prepare bulk data for ingestion
    bulk_data = []
    for packed_ngram in packed_ngrams:
        action = {
            "_op_type": "index",  # Action type for bulk request
            "_index": "packed_integers_index",  # Updated to use the new index name
            "_source": {
                "packed_ngram": packed_ngram
            }
        }
        bulk_data.append(action)
    
    # Perform bulk insert into OpenSearch Serverless (AOSS)
    success, failed = bulk(opensearch_client, bulk_data)
    
    print(f"Ingested {success} n-grams successfully!")
    if failed > 0:
        print(f"Failed to ingest {failed} n-grams.")

# 4. Search for a specific 4-byte n-gram (packaged as an integer)
def search_ngram(query_ngram):
    # Convert query n-gram to packed integer
    query_int = struct.unpack(">Q", query_ngram.ljust(8, b'\0'))[0]
    
    # Search query body
    query_body = {
        "query": {
            "match": {
                "packed_ngram": query_int
            }
        }
    }

    # Perform search on OpenSearch Serverless
    response = opensearch_client.search(
        index="packed_integers_index",  # Search on the updated index name
        body=query_body
    )

    if response['hits']['total']['value'] > 0:
        print(f"Search results for n-gram {query_ngram}: {response['hits']['hits']}")
    else:
        print(f"N-gram {query_ngram} not found.")

# Main execution
if __name__ == "__main__":
    # Create the OpenSearch index (only run this once to set up the index)
    # create_index()

    # Ingest the 5GB binary file's n-grams (uncomment the line below when ready)
    ingest_ngrams("/Users/kienpham/aws/bin-db/data/random_binary_file.bin")

    # Search for a random 4-byte n-gram (replace with your own sample n-gram)
    # random_ngram = b"abcd"  # Replace with your sample 4-byte n-gram
    # search_ngram(random_ngram)