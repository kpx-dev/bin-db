import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import json
from typing import List, Generator, Dict, Any
import binascii
import os
from dotenv import load_dotenv

class BinDB:
    def __init__(self, index_name: str = "binary-ngrams"):        
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

    @property
    def _index_settings(self) -> Dict[str, Any]:
        """Define index settings for 4-byte ngrams"""
        return {
            "index.max_ngram_diff": 0,
            "analysis": {
                "analyzer": {
                    "ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "ngram_tokenizer"
                    }
                },
                "tokenizer": {
                    "ngram_tokenizer": {
                        "type": "ngram",
                        "min_gram": 8,  # 4 bytes = 8 hex characters
                        "max_gram": 8,
                        "token_chars": []
                    }
                }
            }
        }

    @property
    def _index_mappings(self) -> Dict[str, Any]:
        """Define index mappings"""
        return {
            "properties": {
                "file_path": {"type": "keyword"},
                "file_sha256": {"type": "keyword"},
                "offset": {"type": "long"},
                "hex_ngram": {
                    "type": "text",
                    "analyzer": "ngram_analyzer",
                    "search_analyzer": "standard"
                }
            }
        }

    def create_index(self) -> None:
        """Create the index with appropriate mappings for binary n-grams"""
        if not self.client.indices.exists(self.index_name):
            mapping = {
                "settings": self._index_settings,
                "mappings": self._index_mappings
            }
            self.client.indices.create(index=self.index_name, body=mapping)
            print("Created index with mapping:", json.dumps(mapping, indent=2))

    def generate_ngrams(self, binary_data: bytes) -> Generator[tuple, None, None]:
        """Generate 4-byte ngrams from binary data with step size 1"""
        data_length = len(binary_data)
        for offset in range(data_length - 3):  # -3 to ensure 4 bytes
            ngram = binary_data[offset:offset + 4]
            yield (ngram, offset)

    def index_binary_file(self, file_path: str, file_sha256: str, batch_size: int = 1000) -> None:
        """Index a binary file into OpenSearch"""
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
            # Convert 4 bytes to hex
            hex_ngram = binascii.hexlify(ngram).decode('ascii')
            
            doc = {
                "file_path": file_path,
                "file_sha256": file_sha256,
                "offset": offset,
                "hex_ngram": hex_ngram
            }
            
            bulk_data.extend([
                {"index": {"_index": self.index_name}},
                doc
            ])
            
            processed_ngrams += 1
            
            if len(bulk_data) >= batch_size * 2:
                res = self.client.bulk(body=bulk_data)
                if not res.get('errors', False):
                    successful_docs += len(bulk_data) // 2
                
                # Print progress
                progress = (processed_ngrams / total_ngrams) * 100
                print(f"\rProgress: {processed_ngrams:,}/{total_ngrams:,} ngrams "
                      f"({progress:.2f}%) - Successfully indexed: {successful_docs:,} docs", 
                      end='', flush=True)
                
                bulk_data = []
        
        # Process remaining documents
        if bulk_data:
            res = self.client.bulk(body=bulk_data)
            if not res.get('errors', False):
                successful_docs += len(bulk_data) // 2
        
        print(f"\nIndexing complete!")
        print(f"Total ngrams processed: {processed_ngrams:,}")
        print(f"Successfully indexed documents: {successful_docs:,}")

    def search_by_ngram(self, ngram: bytes) -> List[Dict[str, Any]]:
        """Search for documents containing the specified 4-byte ngram"""
        try:
            # Convert 4 bytes to hex
            hex_ngram = binascii.hexlify(ngram).decode('ascii')
            print(f"Searching for hex ngram: {hex_ngram}")

            query = {
                "query": {
                    "match": {
                        "hex_ngram": hex_ngram
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
    # Initialize BinDB
    bindb = BinDB()
    
    # Create index
    bindb.create_index()
    
    # Index sample file
    file_path = "data/random_binary_file_10MB.bin"
    file_sha256 = "sample_sha256"  # In real usage, calculate actual SHA256
    
    print("\nIndexing file...")
    bindb.index_binary_file(file_path, file_sha256)
    
    # Sample searches
    print("\nPerforming sample searches...")
    
    # Search for first 4 bytes of file
    with open(file_path, 'rb') as f:
        first_ngram = f.read(4)
    print("\nSearching for first 4 bytes of file:")
    results = bindb.search_by_ngram(first_ngram)
    print(f"Found {len(results)} matches")
    for result in results:
        print(f"File: {result['file_path']}, Offset: {result['offset']}")
    
    # Search for a specific pattern
    test_ngram = bytes([0xDE, 0xAD, 0xBE, 0xEF])  # DEADBEEF
    print("\nSearching for DEADBEEF pattern:")
    results = bindb.search_by_ngram(test_ngram)
    print(f"Found {len(results)} matches")
    for result in results:
        print(f"File: {result['file_path']}, Offset: {result['offset']}")