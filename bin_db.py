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
            "index.max_ngram_diff": 8,  # Allow larger difference between min and max gram
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
                        "min_gram": 4,  # Minimum n-gram size
                        "max_gram": 8,  # Maximum n-gram size
                        "token_chars": []  # Empty array means tokenize everything
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
                "is_binary": {"type": "boolean"},
                "content": {
                    "type": "text",
                    "analyzer": "ngram_analyzer",
                    "search_analyzer": "standard",
                    "term_vector": "with_positions_offsets_payloads",
                    "index_options": "positions"
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
        else:
            # Get and print current mapping
            current_mapping = self.client.indices.get_mapping(index=self.index_name)
            print("Current index mapping:", json.dumps(current_mapping, indent=2))

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
        """Index an entire file into OpenSearch with automatic n-gram tokenization"""
        print("Indexing entire file...")
        is_binary_file = self.is_binary_file(file_path)
        print(f"Is binary file: {is_binary_file}")
        
        # Read file content
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        
        # For binary files, convert each byte to hex representation
        # For text files, decode as UTF-8
        if is_binary_file:
            hex_content = ''
            for byte in binary_data:
                hex_content += f'{byte:02x}'
            content = hex_content
        else:
            content = binary_data.decode('utf-8', errors='ignore')
        
        # Prepare the document
        doc = {
            "file_path": file_path,
            "file_sha256": file_sha256,
            "content": content,
            "is_binary": is_binary_file
        }

        print(f"Indexing document with ID (SHA256): {file_sha256}")
        print("Document content length:", len(content))

        if not dry_run:
            try:
                # Use file_sha256 as document ID
                res = self.client.index(
                    index=self.index_name,
                    id=file_sha256,
                    body=doc
                )
                print('Indexing result:', json.dumps(res, indent=2))

                # Refresh the index to make the document searchable
                self.client.indices.refresh(index=self.index_name)

                # Verify the document was indexed
                try:
                    verify = self.client.get(
                        index=self.index_name,
                        id=file_sha256
                    )
                    print("Document verification:", json.dumps(verify, indent=2))
                except Exception as e:
                    print(f"Error verifying document: {str(e)}")

            except Exception as e:
                print(f"Error indexing file: {str(e)}")
        
        print("Indexing complete!")

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

    def search_by_ngram(self, ngram: str, size: int = 2) -> List[Dict[str, Any]]:
        """
        Search for documents containing the specified ngram
        """
        try:
            # If the search term is binary, convert it to hex
            if isinstance(ngram, bytes):
                search_term = binascii.hexlify(ngram).decode('ascii')
            else:
                search_term = ngram

            print(f"Searching for term: {search_term}")

            # First, perform the search
            query = {
                "query": {
                    "match_phrase": {
                        "content": search_term
                    }
                },
                "size": size,
                "_source": ["file_path", "file_sha256", "is_binary"],
                "term_statistics": True,
                "field_statistics": True,
                "positions": True,
                "offsets": True,
                "filter": {
                    "max_num_terms": 100
                }
            }

            print("Search query:", json.dumps(query, indent=2))

            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            print("Search response:", json.dumps(response, indent=2))
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                doc_id = hit['_id']
                
                # Get term vectors for this document
                tv_response = self.client.termvectors(
                    index=self.index_name,
                    id=doc_id,
                    fields=['content'],
                    positions=True,
                    offsets=True,
                    payloads=True,
                    term_statistics=True,
                    field_statistics=True,
                    body={
                        "fields": ["content"],
                        "offsets": True,
                        "positions": True,
                        "term_statistics": True,
                        "field_statistics": True
                    }
                )
                
                print("Term vectors response:", json.dumps(tv_response, indent=2))
                
                # Extract matches from term vectors
                matches = []
                if ('term_vectors' in tv_response and 
                    'content' in tv_response['term_vectors'] and 
                    'terms' in tv_response['term_vectors']['content']):
                    
                    terms = tv_response['term_vectors']['content']['terms']
                    for term, term_info in terms.items():
                        if 'tokens' in term_info:
                            for token in term_info['tokens']:
                                matches.append({
                                    'start_offset': token.get('start_offset'),
                                    'end_offset': token.get('end_offset'),
                                    'position': token.get('position'),
                                    'value': term
                                })

                result = {
                    'file_path': source['file_path'],
                    'file_sha256': source['file_sha256'],
                    'score': hit['_score'],
                    'matches': matches,
                    'is_binary': source.get('is_binary', False)
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            print(e)
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