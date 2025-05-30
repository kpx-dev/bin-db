import json
import struct
import boto3
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from requests_aws4auth import AWS4Auth
from typing import List, Generator, Dict, Any

class PackedIntegersDB:
    def __init__(self, index_name: str = "packed-integers"):
        self.index_name = index_name
        self.client = self._initialize_client()

    def _initialize_client(self) -> OpenSearch:
        """Initialize and return OpenSearch client with AWS authentication"""
        load_dotenv()
        endpoint = os.getenv('OPENSEARCH_ENDPOINT')
        if not endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT not found in .env file")
            
        credentials = boto3.Session().get_credentials()
        region = endpoint.split('.')[1]  # Extract region from endpoint
        
        aws_auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            'es',
            session_token=credentials.token
        )
        
        return OpenSearch(
            hosts=[{'host': endpoint.replace('https://', ''), 'port': 443}],
            http_auth=aws_auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=10000
        )

    def create_index(self) -> None:
        """Create the index with appropriate mappings for packed integers"""
        index_definition = {
            "settings": {
                "analysis": {
                    "tokenizer": {
                        "ngram_tokenizer": {
                            "type": "ngram",
                            "min_gram": 4,
                            "max_gram": 4
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
                    "file_path": {"type": "keyword"},
                    "file_sha256": {"type": "keyword"},
                    "offset": {"type": "long"},
                    "packed_ngram": {
                        "type": "long"  # Store 4-byte ngrams as 64-bit integers
                    }
                }
            }
        }

        if not self.client.indices.exists(self.index_name):
            response = self.client.indices.create(
                index=self.index_name,
                body=index_definition
            )
            if response.get('acknowledged', False):
                print("Index created successfully!")
            else:
                print(f"Error creating index: {response}")

    def generate_ngrams(self, binary_data: bytes) -> Generator[tuple, None, None]:
        """Generate 4-byte ngrams from binary data with step size 1"""
        for offset in range(len(binary_data) - 3):  # -3 to ensure 4 bytes
            ngram = binary_data[offset:offset + 4]
            yield (ngram, offset)

    def pack_ngram(self, ngram: bytes) -> int:
        """Convert 4-byte ngram to packed 64-bit integer"""
        return struct.unpack(">Q", ngram.ljust(8, b'\0'))[0]

    def unpack_ngram(self, packed_int: int) -> bytes:
        """Convert packed 64-bit integer back to 4-byte ngram"""
        return struct.pack(">Q", packed_int)[:4]

    def index_binary_file(self, file_path: str, file_sha256: str, batch_size: int = 1000) -> None:
        """Index a binary file into OpenSearch using packed integers"""
        print(f"\nProcessing file: {file_path}")
        
        # Read file and calculate total ngrams for progress tracking
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        
        total_ngrams = len(binary_data) - 3
        processed_ngrams = 0
        successful_docs = 0
        
        bulk_data = []
        print(f"Total ngrams to process: {total_ngrams:,}")
        
        for ngram, offset in self.generate_ngrams(binary_data):
            # Convert 4 bytes to packed integer
            packed_int = self.pack_ngram(ngram)
            
            doc = {
                "file_path": file_path,
                "file_sha256": file_sha256,
                "offset": offset,
                "packed_ngram": packed_int
            }
            
            bulk_data.extend([
                {"index": {"_index": self.index_name}},
                doc
            ])
            
            processed_ngrams += 1
            
            if len(bulk_data) >= batch_size * 2:
                success, failed = bulk(self.client, bulk_data)
                successful_docs += success
                
                # Print progress
                progress = (processed_ngrams / total_ngrams) * 100
                print(f"\rProgress: {processed_ngrams:,}/{total_ngrams:,} ngrams "
                      f"({progress:.2f}%) - Successfully indexed: {successful_docs:,} docs", 
                      end='', flush=True)
                
                bulk_data = []
        
        # Process remaining documents
        if bulk_data:
            success, failed = bulk(self.client, bulk_data)
            successful_docs += success
        
        print(f"\nIndexing complete!")
        print(f"Total ngrams processed: {processed_ngrams:,}")
        print(f"Successfully indexed documents: {successful_docs:,}")

    def search_by_ngram(self, ngram: bytes) -> List[Dict[str, Any]]:
        """Search for documents containing the specified 4-byte ngram"""
        try:
            # Convert 4 bytes to packed integer
            packed_int = self.pack_ngram(ngram)
            print(f"Searching for packed integer: {packed_int}")

            query = {
                "query": {
                    "term": {
                        "packed_ngram": packed_int
                    }
                },
                "_source": ["file_path", "file_sha256", "offset"]
            }

            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    'file_path': source['file_path'],
                    'file_sha256': source['file_sha256'],
                    'offset': source['offset'],
                    'score': hit['_score']
                })
                
            return results
            
        except Exception as e:
            print(f"Error searching for ngram: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize PackedIntegersDB
    db = PackedIntegersDB()
    
    # Create index
    db.create_index()
    
    # Index sample file
    file_path = "data/random_binary_file_10MB.bin"
    file_sha256 = "sample_sha256"  # In real usage, calculate actual SHA256
    
    print("\nIndexing file...")
    db.index_binary_file(file_path, file_sha256)
    
    # Sample searches
    print("\nPerforming sample searches...")
    
    # Search for first 4 bytes of file
    with open(file_path, 'rb') as f:
        first_ngram = f.read(4)
    print("\nSearching for first 4 bytes of file:")
    results = db.search_by_ngram(first_ngram)
    print(f"Found {len(results)} matches")
    for result in results:
        print(f"File: {result['file_path']}, Offset: {result['offset']}")
    
    # Search for a specific pattern
    test_ngram = bytes([0xDE, 0xAD, 0xBE, 0xEF])  # DEADBEEF
    print("\nSearching for DEADBEEF pattern:")
    results = db.search_by_ngram(test_ngram)
    print(f"Found {len(results)} matches")
    for result in results:
        print(f"File: {result['file_path']}, Offset: {result['offset']}")