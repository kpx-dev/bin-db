import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import os
from typing import List, Generator, Dict, Any
import binascii

class BinDB:
    def __init__(self, collection_endpoint: str, region: str, index_name: str):
        self.index_name = index_name
        self.client = self._initialize_client(collection_endpoint, region)

    def _initialize_client(self, collection_endpoint: str, region: str) -> OpenSearch:
        """Initialize and return OpenSearch client with AWS authentication"""
        credentials = boto3.Session().get_credentials()
        aws_auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            'aoss',
            session_token=credentials.token
        )
        
        return OpenSearch(
            hosts=[{'host': collection_endpoint, 'port': 443}],
            http_auth=aws_auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

    @property
    def _index_settings(self) -> Dict[str, Any]:
        """Define index settings"""
        return {
            "index.max_ngram_diff": 4,
            "analysis": {
                "analyzer": {
                    "ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "ngram_tokenizer",
                        "filter": ["lowercase"]
                    }
                },
                "tokenizer": {
                    "ngram_tokenizer": {
                        "type": "ngram",
                        "min_gram": 4,
                        "max_gram": 8,
                        "token_chars": [
                            "letter",
                            "digit",
                            "punctuation",
                            "symbol"
                        ]
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
                "ngram": {
                    "type": "text",
                    "analyzer": "ngram_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"}
                    }
                },
                "offset": {"type": "long"},
                "size": {"type": "integer"},
                "content": {
                    "type": "text",
                    "analyzer": "ngram_analyzer"
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

    def generate_ngrams(self, binary_data: bytes, min_size: int = 4, 
                       max_size: int = 8) -> Generator[tuple, None, None]:
        """Generate n-grams from binary data"""
        data_length = len(binary_data)
        
        for size in range(min_size, max_size + 1):
            for offset in range(data_length - size + 1):
                ngram = binary_data[offset:offset + size]
                yield (ngram, offset, size)

    def _prepare_document(self, file_path: str, file_sha256: str, 
                         ngram: bytes, offset: int, size: int) -> Dict[str, Any]:
        """Prepare a document for indexing"""
        return {
            "file_path": file_path,
            "file_sha256": file_sha256,
            "ngram": binascii.hexlify(ngram).decode('ascii'),
            "offset": offset,
            "size": size
        }

    def index_binary_file(self, file_path: str, file_sha256: str, min_size: int = 4, 
                         max_size: int = 8, batch_size: int = 1000) -> None:
        """Index a binary file into OpenSearch"""
        with open(file_path, 'rb') as f:
            binary_data = f.read()

        bulk_data = []
        
        for ngram, offset, size in self.generate_ngrams(binary_data, min_size, max_size):
            doc = self._prepare_document(file_path, file_sha256, ngram, offset, size)
            
            bulk_data.extend([
                {"index": {"_index": self.index_name}},
                doc
            ])
            
            if len(bulk_data) >= batch_size * 2:
                res = self.client.bulk(body=bulk_data)
                print("bulk insert res", res)
                bulk_data = []
        
        if bulk_data:
            res = self.client.bulk(body=bulk_data)
            print("bulk insert res", res)

    def delete_index(self) -> None:
        """Delete specific index and its data"""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                print(f"Successfully deleted index: {self.index_name}")
            else:
                print(f"Index {self.index_name} does not exist")
        except Exception as e:
            print(f"Error deleting index: {str(e)}")

    def search_by_ngram(self, ngram: str, size: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents containing the specified ngram
        
        Args:
            ngram (bytes): The binary ngram to search for
            size (int): Maximum number of results to return (default: 10)
            
        Returns:
            List[Dict[str, Any]]: List of matching documents with their details
        """
        try:
            # Convert binary ngram to hex string for searching
            # hex_ngram = binascii.hexlify(ngram).decode('ascii')
            
            # Construct the search query
            query = {
                "query": {
                    "match": {
                        "ngram": ngram
                    }
                },
                "size": size,
                "_source": ["file_path", "file_sha256", "offset", "size"]
            }
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            # Extract and format results
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                results.append(result)
                
            return results
            
        except Exception as e:
            print(f"Error searching for ngram: {str(e)}")
            return []