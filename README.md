# bin-db

Demo repo to show how to do a "full text search" for binary file using 4 bytes ngram.

## Table of Contents
- [Generating Sample Binary Files](#generating-sample-binary-files)
- [Solutions](#solutions)
  - [1. Hex Storage in OpenSearch](#1-hex-storage-in-opensearch)
  - [2. Packed Integers in OpenSearch](#2-packed-integers-in-opensearch)
  - [3. Bloom Filter](#3-bloom-filter)

## Generating Sample Binary Files

To generate sample binary files for testing:

1. Run the generator script from the root directory:
   ```bash
   python utils/generate_test_data.py
   ```

2. The script will:
   - Create a `data` directory if it doesn't exist
   - Generate a 10MB random binary file named `random_binary_file_10MB.bin`
   - Save the file in the `data` directory
   - Display the actual file size after generation

3. To generate a different size file, modify the `size_mb` variable in the script.

## Solutions

### 1. Hex Storage in OpenSearch

This solution stores binary data as hex strings in OpenSearch using ngram tokenizer.

#### Setup
1. Create a `.env` file in the `1-hex-storage-type` directory:
   ```
   OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint.us-east-1.es.amazonaws.com
   ```

2. Install dependencies from the root directory:
   ```bash
   pip install -r requirements.txt
   ```

#### Usage
1. Create the index:
   ```python
   db.create_index()
   ```

2. Index a binary file:
   ```python
   db.index_binary_file("data/random_binary_file_10MB.bin", "file_sha256")
   ```

3. Search for ngrams:
   ```python
   results = db.search_by_ngram(b"\xDE\xAD\xBE\xEF")  # Search for DEADBEEF
   ```

### 2. Packed Integers in OpenSearch

This solution stores binary data as packed 64-bit integers in OpenSearch.

#### Setup
1. Create a `.env` file in the `2-packed-integers-storage-type` directory:
   ```
   OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint.us-east-1.es.amazonaws.com
   ```

2. Install dependencies from the root directory:
   ```bash
   pip install -r requirements.txt
   ```

#### Usage
1. Create the index:
   ```python
   db.create_index()
   ```

2. Index a binary file:
   ```python
   db.index_binary_file("data/random_binary_file_10MB.bin", "file_sha256")
   ```

3. Search for ngrams:
   ```python
   results = db.search_by_ngram(b"\xDE\xAD\xBE\xEF")  # Search for DEADBEEF
   ```

### 3. Bloom Filter

This solution uses a Bloom Filter to efficiently check for the presence of ngrams.

#### Setup
1. Install dependencies from the root directory:
   ```bash
   pip install -r requirements.txt
   ```

#### Usage
1. Process a binary file:
   ```python
   bloom_filter.process_file("data/random_binary_file_10MB.bin")
   ```

2. Check for ngrams:
   ```python
   if bloom_filter.check_ngram(b"\xDE\xAD\xBE\xEF"):
       print("Ngram might be in the file")
   else:
       print("Ngram is definitely not in the file")
   ```

3. Save/Load the Bloom Filter:
   ```python
   # Save
   bloom_filter.save_to_disk("bloom_filter.bin")
   
   # Load
   bloom_filter.load_from_disk("bloom_filter.bin")
   ```