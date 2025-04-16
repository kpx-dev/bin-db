#!/usr/bin/env python3
import os
import base64
import uuid
import json
import time
import argparse
import sys
import concurrent.futures
from datetime import datetime
from functools import partial
import threading
import queue

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers, exceptions
from requests_aws4auth import AWS4Auth

# Configuration - replace with your OpenSearch values
REGION = 'us-east-1'
INDEX_NAME = 'binary_ngram_index'
OPENSEARCH_ENDPOINT = '88e8srnxk6448i46lkp8.us-east-1.aoss.amazonaws.com'

# OpenSearch client setup
def get_opensearch_client():
    """Create and return a properly configured OpenSearch client"""
    try:
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            REGION,
            'aoss',  # Amazon OpenSearch Serverless
            session_token=credentials.token
        )

        client = OpenSearch(
            hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
            retry_on_timeout=True,
            max_retries=3
        )
        return client
    except Exception as e:
        print(f"Warning: OpenSearch client setup failed. You'll need AWS credentials configured: {e}")
        return None

# Create global client for main thread
opensearch_client = get_opensearch_client()

def create_binary_ngram_index():
    """Create an optimized index for storing binary n-grams with minimal storage footprint"""
    
    index_settings = {
        "settings": {
            "index": {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "refresh_interval": "30s",
                "codec": "best_compression",
                "mapping": {
                    "total_fields": {
                        "limit": 2000
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "file_id": {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "chunk_offset": {"type": "long"},
                "ngram": {"type": "binary", "doc_values": False},  # Binary field with doc_values disabled to save space
                "positions": {
                    "type": "integer", 
                    "index": False,        # Don't index positions, just store them
                    "doc_values": True     # Keep doc_values for efficient retrieval
                },
                "frequency": {"type": "integer", "index": False},
                "file_path": {"type": "keyword", "index": True}  # Index for efficient searching by path
            }
        }
    }
    
    # Create the index if it doesn't exist
    if not opensearch_client.indices.exists(index=INDEX_NAME):
        opensearch_client.indices.create(
            index=INDEX_NAME,
            body=index_settings
        )
        print(f"Created index {INDEX_NAME}")
    else:
        print(f"Index {INDEX_NAME} already exists")

def optimize_for_indexing():
    """Configure index for optimal bulk indexing performance"""
    
    opensearch_client.indices.put_settings(
        index=INDEX_NAME,
        body={
            "index": {
                "refresh_interval": "-1",  # Disable refresh during bulk indexing
                "number_of_replicas": 0    # Temporarily remove replicas
            }
        }
    )
    print("Index optimized for bulk indexing")

def restore_search_settings():
    """Restore settings for optimal search performance"""
    
    opensearch_client.indices.put_settings(
        index=INDEX_NAME,
        body={
            "index": {
                "refresh_interval": "30s",  # Re-enable refreshing
                "number_of_replicas": 1     # Restore replicas
            }
        }
    )
    opensearch_client.indices.refresh(index=INDEX_NAME)
    print("Index settings restored for search performance")

# Global progress tracking
class ProgressTracker:
    """Thread-safe progress tracker for parallel processing"""
    def __init__(self, total_chunks):
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        self.docs_indexed = 0
        self.batches_completed = 0
        self.chunks_processed = 0
        self.total_chunks = total_chunks
        self.docs_per_chunk = 0  # To be estimated
        self.last_print_time = time.time()
        self.print_interval = 0.5  # Seconds between progress updates
    
    def update_indexed(self, docs_count, is_chunk_complete=False):
        """Update the indexed document count"""
        with self.lock:
            self.docs_indexed += docs_count
            self.batches_completed += 1
            if is_chunk_complete:
                self.chunks_processed += 1
            
            current_time = time.time()
            if current_time - self.last_print_time > self.print_interval:
                self.print_progress()
                self.last_print_time = current_time
    
    def update_docs_per_chunk(self, count):
        """Update the estimated documents per chunk"""
        with self.lock:
            if self.docs_per_chunk == 0:  # Only set once from first chunk
                self.docs_per_chunk = count
    
    def print_progress(self):
        """Print current progress statistics"""
        elapsed = datetime.now() - self.start_time
        elapsed_seconds = elapsed.total_seconds()
        docs_per_second = self.docs_indexed / elapsed_seconds if elapsed_seconds > 0 else 0
        
        # Calculate estimated completion
        if docs_per_second > 0 and self.docs_per_chunk > 0 and self.total_chunks > 0:
            progress_pct = self.chunks_processed / self.total_chunks * 100
            estimated_docs = self.docs_per_chunk * self.total_chunks
            remaining_docs = estimated_docs - self.docs_indexed
            estimated_seconds = remaining_docs / docs_per_second if docs_per_second > 0 else 0
            estimated_time = time.strftime('%H:%M:%S', time.gmtime(estimated_seconds))
            
            print(f"\rProgress: {progress_pct:.1f}% | Chunks: {self.chunks_processed}/{self.total_chunks} | " +
                  f"Docs: {self.docs_indexed:,} | {docs_per_second:.1f} docs/sec | " + 
                  f"Est. remaining: {estimated_time}", end='')
            sys.stdout.flush()
        else:
            print(f"\rChunks: {self.chunks_processed}/{self.total_chunks} | " +
                  f"Docs: {self.docs_indexed:,} | {docs_per_second:.1f} docs/sec", end='')
            sys.stdout.flush()

def index_bulk_data(bulk_data, progress_tracker, client=None):
    """
    Index a batch of documents using bulk API
    
    Args:
        bulk_data: Data for bulk indexing
        progress_tracker: Thread-safe progress tracking object
        client: OpenSearch client (thread-specific)
    """
    if not bulk_data:
        return
    
    # Use thread-specific client or global client
    es = client if client is not None else opensearch_client
    
    # Try up to 3 times with exponential backoff
    retries = 0
    max_retries = 3
    retry_delay = 1  # Initial delay in seconds
    
    while retries <= max_retries:
        try:
            response = es.bulk(body="\n".join(map(json.dumps, bulk_data)) + "\n")
            
            # Check for errors
            if response.get('errors', False):
                error_count = sum(1 for item in response['items'] if 'error' in item['index'])
                if error_count > 0:
                    print(f"\nWarning: Bulk indexing completed with {error_count} errors")
            
            # Update progress tracking
            progress_tracker.update_indexed(len(bulk_data) // 2)
            return True
            
        except exceptions.ConnectionTimeout:
            retries += 1
            if retries <= max_retries:
                sleep_time = retry_delay * (2 ** (retries - 1))  # Exponential backoff
                print(f"\nConnection timeout. Retrying in {sleep_time} seconds (attempt {retries}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                print("\nFailed after max retries. Skipping batch.")
                return False
        
        except Exception as e:
            print(f"\nError in bulk indexing: {str(e)}")
            return False

def process_chunk(chunk_info, progress_tracker):
    """
    Process a single chunk from the file
    
    Args:
        chunk_info: Dictionary with chunk information
        progress_tracker: Thread-safe progress tracking object
        
    Returns:
        Number of documents indexed
    """
    chunk_id = chunk_info['chunk_id']
    chunk_offset = chunk_info['chunk_offset']
    chunk_size = chunk_info['chunk_size']
    file_path = chunk_info['file_path']
    file_id = chunk_info['file_id']
    
    # Create thread-local OpenSearch client
    client = get_opensearch_client()
    
    # Read the chunk data
    with open(file_path, 'rb') as f:
        f.seek(chunk_offset)
        chunk_data = f.read(chunk_size)
    
    if not chunk_data:
        return 0
    
    # Dictionary to store n-gram occurrences
    ngram_dict = {}
    
    # Extract 4-byte n-grams with step size 1
    for i in range(len(chunk_data) - 3):
        ngram = chunk_data[i:i+4]
        
        # Use ngram bytes as dictionary key
        if ngram not in ngram_dict:
            ngram_dict[ngram] = []
        
        # Store position within the chunk
        ngram_dict[ngram].append(i)
    
    # Update progress tracker with docs per chunk estimate
    progress_tracker.update_docs_per_chunk(len(ngram_dict))
    
    # Prepare documents for bulk indexing
    bulk_data = []
    docs_indexed = 0
    
    for ngram, positions in ngram_dict.items():
        # Base64 encode the ngram for storage - most efficient representation
        ngram_b64 = base64.b64encode(ngram).decode()
        
        doc = {
            "file_id": file_id,
            "chunk_id": chunk_id,
            "chunk_offset": chunk_offset,
            "ngram": ngram_b64,
            "positions": positions,
            "frequency": len(positions),
            "file_path": file_path
        }
        
        # Add to bulk operations
        bulk_data.append({"index": {"_index": INDEX_NAME}})
        bulk_data.append(doc)
        
        # Index in batches to avoid request size limits
        if len(bulk_data) >= 2000:
            if index_bulk_data(bulk_data, progress_tracker, client):
                docs_indexed += len(bulk_data) // 2
            bulk_data = []
    
    # Index any remaining documents
    if bulk_data:
        if index_bulk_data(bulk_data, progress_tracker, client):
            docs_indexed += len(bulk_data) // 2
    
    # Mark chunk as complete
    progress_tracker.update_indexed(0, is_chunk_complete=True)
    
    return docs_indexed

def process_binary_file_parallel(file_path, file_id=None, chunk_size=10_485_760, max_workers=None):
    """
    Process a binary file in parallel using multiple workers
    
    Args:
        file_path: Path to the local binary file
        file_id: Optional UUID for the file (generated if not provided)
        chunk_size: Size of chunks to process (default: 10MB)
        max_workers: Maximum number of workers (default: CPU count)
    """
    if not file_id:
        file_id = str(uuid.uuid4())
    
    # Get file size
    file_size = os.path.getsize(file_path)
    total_chunks = (file_size + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"Processing file {file_path} ({file_size/1024/1024:.2f} MB) with ID {file_id}")
    print(f"Splitting into {total_chunks} chunks of {chunk_size/1024/1024:.2f} MB each")
    
    # Determine optimal number of workers
    if not max_workers:
        # Use CPU count, but with a reasonable upper limit
        max_workers = min(os.cpu_count(), 16)
    
    # Also limit by number of chunks
    max_workers = min(max_workers, total_chunks)
    # max_workers = 20
    print(f"Using {max_workers} parallel workers")
    
    # Initialize progress tracking
    progress_tracker = ProgressTracker(total_chunks)
    
    # Prepare chunk information
    chunks = []
    for chunk_id in range(total_chunks):
        chunk_offset = chunk_id * chunk_size
        current_chunk_size = min(chunk_size, file_size - chunk_offset)
        
        chunks.append({
            'chunk_id': chunk_id,
            'chunk_offset': chunk_offset,
            'chunk_size': current_chunk_size,
            'file_path': file_path,
            'file_id': file_id
        })
    
    # Process chunks in parallel
    start_time = time.time()
    total_docs = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_chunk, chunk, progress_tracker): chunk
            for chunk in chunks
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                docs_indexed = future.result()
                total_docs += docs_indexed
            except Exception as exc:
                print(f"\nChunk {chunk['chunk_id']} generated an exception: {exc}")
    
    # Final progress update
    progress_tracker.print_progress()
    
    # Summarize processing results
    elapsed = time.time() - start_time
    elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    
    print(f"\n\nCompleted processing file {file_path}")
    print(f"- Chunks processed: {total_chunks}")
    print(f"- Documents indexed: {progress_tracker.docs_indexed:,}")
    print(f"- Processing time: {elapsed_str}")
    print(f"- Average indexing speed: {progress_tracker.docs_indexed/elapsed:.1f} docs/sec")
    
    return file_id

def search_binary_pattern(pattern, max_results=100):
    """
    Search for a binary pattern across indexed files
    
    Args:
        pattern: Bytes object containing the pattern to search
        max_results: Maximum number of results to return
    
    Returns:
        List of matches with file and position information
    """
    if len(pattern) < 4:
        raise ValueError("Search pattern must be at least 4 bytes")
    
    # Extract the first n-gram from the pattern
    first_ngram = pattern[0:4]
    target_ngram_b64 = base64.b64encode(first_ngram).decode()
    
    print(f"Searching for pattern starting with bytes: {first_ngram.hex()}")
    
    # Search for the target n-gram
    query = {
        "size": max_results,
        "query": {
            "term": {
                "ngram": target_ngram_b64
            }
        }
    }
    
    response = opensearch_client.search(index=INDEX_NAME, body=query)
    
    matches = []
    for hit in response['hits']['hits']:
        doc = hit['_source']
        file_id = doc['file_id']
        chunk_offset = doc['chunk_offset']
        file_path = doc['file_path']
        
        # For each position where this n-gram occurs
        for pos in doc['positions']:
            file_pos = chunk_offset + pos
            
            # For longer patterns (> 4 bytes), we could verify the full pattern match
            # For now, return potential matches based on the first n-gram
            matches.append({
                "file_id": file_id,
                "position": file_pos,
                "file_path": file_path,
                "score": hit['_score']
            })
    
    print(f"Found {len(matches)} potential matches")
    return matches

def retrieve_binary_content(file_id, position, length, buffer=100):
    """
    Retrieve binary content from a local file based on a search match
    
    Args:
        file_id: File ID
        position: Starting byte position
        length: Number of bytes to retrieve
        buffer: Additional bytes to retrieve around the match
    
    Returns:
        Binary content from file
    """
    # Find the file path
    query = {
        "size": 1,
        "query": {
            "term": {
                "file_id": file_id
            }
        }
    }
    
    response = opensearch_client.search(index=INDEX_NAME, body=query)
    
    if response['hits']['total']['value'] == 0:
        raise ValueError(f"File ID {file_id} not found")
    
    file_path = response['hits']['hits'][0]['_source']['file_path']
    
    # Calculate range with buffer
    start = max(0, position - buffer)
    end = position + length + buffer
    
    # Read from file
    with open(file_path, 'rb') as f:
        f.seek(start)
        content = f.read(end - start)
    
    # Calculate where the actual match starts in our buffered content
    match_start = position - start
    match_end = match_start + length
    
    return {
        "full_content": content,
        "matched_content": content[match_start:match_end],
        "position": position,
        "file_path": file_path
    }

def analyze_storage_usage():
    """Analyze storage usage of the index"""
    
    # Get index statistics
    stats = opensearch_client.indices.stats(index=INDEX_NAME)
    
    # Calculate storage metrics
    primary_size_bytes = stats['_all']['primaries']['store']['size_in_bytes']
    primary_size_mb = primary_size_bytes / 1024 / 1024
    
    total_size_bytes = stats['_all']['total']['store']['size_in_bytes']
    total_size_mb = total_size_bytes / 1024 / 1024
    
    doc_count = stats['_all']['primaries']['docs']['count']
    
    print(f"Index Statistics for {INDEX_NAME}:")
    print(f"Document count: {doc_count:,}")
    print(f"Primary shards size: {primary_size_mb:.2f} MB")
    print(f"Total size (with replicas): {total_size_mb:.2f} MB")
    
    # Calculate storage ratio against known indexed files
    query = {
        "size": 0,
        "aggs": {
            "files": {
                "terms": {
                    "field": "file_path",
                    "size": 1000
                }
            }
        }
    }
    
    response = opensearch_client.search(index=INDEX_NAME, body=query)
    
    total_file_size = 0
    print("\nIndexed Files:")
    for bucket in response['aggregations']['files']['buckets']:
        file_path = bucket['key']
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_file_size += file_size
            print(f"- {file_path}, Size: {file_size/1024/1024:.2f} MB, Doc Count: {bucket['doc_count']:,}")
    
    if total_file_size > 0:
        ratio = primary_size_mb / (total_file_size/1024/1024)
        print(f"\nIndex is approximately {ratio:.2%} the size of original files")
        
        # Print optimization suggestion if the index is larger than original files
        if ratio > 1.0:
            print("\nStorage Optimization Suggestions:")
            print("1. Increase the chunk size to reduce document overhead")
            print("2. Consider increasing the n-gram size (would require reindexing)")
            print("3. For very large files, consider indexing only a subset of n-grams")
            print("4. Force merge the index to reclaim deleted space: opensearch_client.indices.forcemerge(index=INDEX_NAME, max_num_segments=1)")

def cleanup_resources(file_id=None):
    """
    Clean up resources created by the process
    
    Args:
        file_id: Optional file ID to delete only specific documents
    """
    # Delete documents from OpenSearch
    if file_id:
        delete_query = {
            "query": {
                "term": {
                    "file_id": file_id
                }
            }
        }
        opensearch_client.delete_by_query(
            index=INDEX_NAME,
            body=delete_query
        )
        print(f"Deleted all documents for file_id {file_id}")
    else:
        # Be careful with this!
        if opensearch_client.indices.exists(index=INDEX_NAME):
            opensearch_client.indices.delete(index=INDEX_NAME)
            print(f"Deleted entire index {INDEX_NAME}")

def main():
    parser = argparse.ArgumentParser(description='Binary File Indexing and Search for Amazon OpenSearch - Parallel Edition')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate file command
    generate_parser = subparsers.add_parser('generate', help='Generate a random binary file')
    generate_parser.add_argument('file_path', help='Path where to create the file')
    generate_parser.add_argument('--size', type=float, default=5.0, help='Size of the file in GB')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Create the index')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a binary file')
    index_parser.add_argument('file_path', help='Path to binary file to index')
    index_parser.add_argument('--chunk-size', type=int, default=10_485_760, help='Chunk size in bytes (default: 10MB)')
    index_parser.add_argument('--workers', type=int, help='Maximum number of parallel workers')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for a binary pattern')
    search_parser.add_argument('pattern', help='Binary pattern to search for (hex string)')
    search_parser.add_argument('--results', type=int, default=10, help='Maximum results to return')
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve binary content')
    retrieve_parser.add_argument('file_id', help='File ID')
    retrieve_parser.add_argument('position', type=int, help='Starting byte position')
    retrieve_parser.add_argument('length', type=int, help='Number of bytes to retrieve')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show index statistics')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up resources')
    cleanup_parser.add_argument('--file-id', help='Specific file ID to clean up')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_random_binary_file(args.file_path, args.size)
    
    elif args.command == 'setup':
        if not opensearch_client:
            print("Error: OpenSearch client not configured. Check AWS credentials.")
            return
        create_binary_ngram_index()
    
    elif args.command == 'index':
        if not opensearch_client:
            print("Error: OpenSearch client not configured. Check AWS credentials.")
            return
        create_binary_ngram_index()  # Ensure index exists
        # optimize_for_indexing()
        # try:
        file_id = process_binary_file_parallel(args.file_path, chunk_size=args.chunk_size, max_workers=args.workers)
        print(f"File indexed with ID: {file_id}")
        # finally:
            # restore_search_settings()
    
    elif args.command == 'search':
        if not opensearch_client:
            print("Error: OpenSearch client not configured. Check AWS credentials.")
            return
        # Convert hex string to bytes
        pattern_bytes = bytes.fromhex(args.pattern)
        matches = search_binary_pattern(pattern_bytes, args.results)
        
        print("\nSearch Results:")
        for i, match in enumerate(matches[:args.results]):
            print(f"{i+1}. File: {match['file_id']}, Position: {match['position']:,}")
            print(f"   Path: {match['file_path']}")
    
    elif args.command == 'retrieve':
        if not opensearch_client:
            print("Error: OpenSearch client not configured. Check AWS credentials.")
            return
        content = retrieve_binary_content(args.file_id, args.position, args.length)
        print("\nRetrieved Content:")
        print(f"Position: {content['position']:,}")
        print(f"Matched content (hex): {content['matched_content'].hex()}")
        print(f"File Path: {content['file_path']}")
    
    elif args.command == 'stats':
        if not opensearch_client:
            print("Error: OpenSearch client not configured. Check AWS credentials.")
            return
        analyze_storage_usage()
    
    elif args.command == 'cleanup':
        if not opensearch_client:
            print("Error: OpenSearch client not configured. Check AWS credentials.")
            return
        cleanup_resources(args.file_id)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


# # Create the OpenSearch index
# python binary_indexer.py setup

# # Index the 5GB binary file
# python binary_indexer.py index test_data.bin

# # Search for a specific hex pattern
# python binary_indexer.py search "DEADBEEF"

# # Retrieve 100 bytes at a specific position
# python binary_indexer.py retrieve abc123def 1000000 100

# # Show storage statistics
# python binary_indexer.py stats

# # Clean up everything
# python binary_indexer.py cleanup