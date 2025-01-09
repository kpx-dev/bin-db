import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import os
import json
from typing import List, Generator, Dict, Any
import binascii
from botocore.config import Config

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
            connection_class=RequestsHttpConnection,
            timeout=10000,  # Timeout in seconds,
            max_retries=3,
            retry_on_timeout=True
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
        
        index_property = {
            "type": "text",
            "analyzer": "ngram_analyzer",
            "fields": {
                "raw": {"type": "keyword"}
            }
        }
        payload = {
            "properties": {
                "file_path": {"type": "keyword"},
                "file_sha256": {"type": "keyword"},
                "offset": {"type": "long"},
                "size": {"type": "integer"},
                "content": {
                    "type": "text",
                    "analyzer": "ngram_analyzer"
                }
            }
        }
        payload['properties'][self.index_name] = index_property

        return payload

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

    def _prepare_document(self, is_binary_file: bool, file_path: str, file_sha256: str, 
                         ngram: bytes, offset: int, size: int) -> Dict[str, Any]:
        """Prepare a document for indexing"""
        body = binascii.hexlify(ngram).decode('ascii') if is_binary_file else str(ngram),
        payload = {
            "file_path": file_path,
            "file_sha256": file_sha256,
            "offset": offset,
            "size": size
        }
        payload[self.index_name] = body

        return payload

    def index_entire_file(self, file_path: str, file_sha256: str, min_size: int = 4, 
                         max_size: int = 8, batch_size: int = 1000, dry_run: bool = False) -> None:
        print("index_entire_file")
        is_binary_file = self.is_binary_file(file_path)
        print(f"Is binary file: {is_binary_file}")
        
        # Read file and calculate total ngrams for progress tracking
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        
        bulk_data = []
        offset = 0
        size = 0
        doc = self._prepare_document(is_binary_file, file_path, file_sha256, binary_data, offset, size)
            
        bulk_data.extend([
            {"index": {"_index": self.index_name}},
            doc
        ])

        res = self.client.bulk(body=bulk_data)
        
        # Print final statistics
        print(f"\nIndexing complete!")

    def index_binary_file(self, file_path: str, file_sha256: str, min_size: int = 4, 
                         max_size: int = 8, batch_size: int = 1000, dry_run: bool = False) -> None:
        """Index a binary file into OpenSearch"""
        is_binary_file = self.is_binary_file(file_path)
        print(f"Is binary file: {is_binary_file}")
        
        # Read file and calculate total ngrams for progress tracking
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        
        data_length = len(binary_data)
        total_ngrams = sum(data_length - size + 1 for size in range(min_size, max_size + 1))
        processed_ngrams = 0
        successful_docs = 0
        
        bulk_data = []
        print(f"\nProcessing file: {file_path}")
        print(f"Total ngrams to process: {total_ngrams:,}")
        
        for ngram, offset, size in self.generate_ngrams(binary_data, min_size, max_size):
            doc = self._prepare_document(is_binary_file, file_path, file_sha256, ngram, offset, size)
            
            bulk_data.extend([
                {"index": {"_index": self.index_name}},
                doc
            ])
            
            processed_ngrams += 1
            
            if len(bulk_data) >= batch_size * 2:
                if not dry_run:
                    res = self.client.bulk(body=bulk_data)
                
                    # Count successful operations
                    if not res.get('errors', False):
                        successful_docs += len(bulk_data) // 2
                
                # Print progress
                progress = (processed_ngrams / total_ngrams) * 100
                print(f"\rProgress: {processed_ngrams:,}/{total_ngrams:,} ngrams "
                      f"({progress:.2f}%) - Successfully indexed: {successful_docs:,} docs", 
                      end='', flush=True)
                
                bulk_data = []
        
        # Process remaining documents
        if bulk_data and not dry_run:
            res = self.client.bulk(body=bulk_data)
            if not res.get('errors', False):
                successful_docs += len(bulk_data) // 2
        
        # Print final statistics
        print(f"\nIndexing complete!")
        print(f"Total ngrams processed: {processed_ngrams:,}")
        print(f"Successfully indexed documents: {successful_docs:,}")

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
            size (int): Maximum number of results to return (default: 3)
            
        Returns:
            List[Dict[str, Any]]: List of matching documents with their details
        """
        try:
            # Convert binary ngram to hex string for searching
            # hex_ngram = binascii.hexlify(ngram).decode('ascii')
            
            # Construct the search query
            match = {}
            match[self.index_name] = ngram
            query = {
                "query": {
                    "match": match
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

    def is_binary_file(self, file_path: str, sample_size: int = 8192) -> bool:
        """
        Detect if a file is binary or text based on its content
        
        Args:
            file_path: Path to the file to check
            sample_size: Number of bytes to check (default: 8KB)
            
        Returns:
            bool: True if file appears to be binary, False if it appears to be text
        """
        # Common binary file signatures (magic numbers)
        BINARY_SIGNATURES = {
            b'\x7fELF',  # ELF files
            b'MZ',       # DOS/PE files
            b'\x89PNG',  # PNG images
            b'PK',       # ZIP files
            b'\xff\xd8', # JPEG files
            b'%PDF',     # PDF files
        }
        
        try:
            with open(file_path, 'rb') as f:
                # Check first few bytes against known binary signatures
                initial_bytes = f.read(4)
                for signature in BINARY_SIGNATURES:
                    if initial_bytes.startswith(signature):
                        return True
                
                # Reset file pointer
                f.seek(0)
                # Read a chunk of the file
                chunk = f.read(sample_size)
                
                # Check for NULL bytes and high number of non-printable characters
                null_count = chunk.count(b'\x00')
                if null_count > 0:
                    return True
                    
                # Check if the chunk contains a high ratio of non-printable characters
                # Excluding common text file characters like newlines and tabs
                control_chars = sum(1 for byte in chunk 
                                  if byte < 8 or byte > 13 and byte < 32 or byte > 126)
                
                # If more than 30% non-printable characters, likely binary
                return (control_chars / len(chunk)) > 0.30
                
        except Exception as e:
            print(f"Error checking file type: {str(e)}")
            return False