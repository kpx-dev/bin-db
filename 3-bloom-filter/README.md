# 4-Byte Ngram Bloom Filter

This implementation provides an efficient way to store and query 4-byte ngrams from binary files using a Bloom Filter with minimal false positives.

## Features

- Efficient storage of 4-byte ngrams using a Bloom Filter
- Uses MurmurHash3 and SHA-256 for high-quality hashing
- Multiple hash functions to reduce false positives
- Automatic calculation of optimal Bloom Filter size
- Memory-efficient implementation using numpy arrays
- Progress tracking during file processing
- Save/Load functionality to persist Bloom Filters
- Secondary verification set for high-confidence items
- Very low false positive rate (configurable, default 0.01%)

## Requirements

- Python 3.7+
- Dependencies listed in requirements.txt

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from ngram_bloom_filter import BloomFilter, calculate_optimal_size, process_binary_file

# Calculate optimal size for your file with 0.01% false positive rate
file_path = "your_file.bin"
false_positive_rate = 0.0001  # 0.01%
size, num_hash_functions = calculate_optimal_size(file_size, false_positive_rate)

# Create Bloom Filter
bloom_filter = BloomFilter(size, num_hash_functions)

# Process a binary file (shows progress)
process_binary_file("your_file.bin", bloom_filter)

# Save the Bloom Filter for later use
bloom_filter.save("data/bloom_filter.npz")

# Load a previously saved Bloom Filter
loaded_bloom_filter = BloomFilter.load("data/bloom_filter.npz")

# Check if specific 4-byte sequences exist
test_ngram = b"\x00\x01\x02\x03"
if test_ngram in bloom_filter:
    print("Ngram might be in the file")
else:
    print("Ngram is definitely not in the file")
```

## How it Works

1. The Bloom Filter uses multiple hash functions (MurmurHash3 and SHA-256) to map each 4-byte ngram to multiple positions in a bit array.
2. When checking if an ngram exists, it checks all corresponding positions in the bit array.
3. If any position is 0, the ngram is definitely not in the set.
4. If all positions are 1, the ngram might be in the set (with a very small probability of false positive).
5. High-confidence items are stored in a secondary verification set for 100% accuracy.

## Performance Considerations

- The false positive rate can be controlled by adjusting the size of the Bloom Filter and the number of hash functions.
- The implementation uses numpy arrays for efficient bit storage.
- MurmurHash3 and SHA-256 provide fast and high-quality hashing suitable for this use case.
- Progress tracking shows real-time processing status.
- Save/Load functionality allows reusing Bloom Filters without reprocessing files.

## Example Output

```
File contains 10,485,757 possible 4-byte ngrams
Bloom Filter size: 251,658,240 bits
Number of hash functions: 17

Processing file: data/random_binary_file_10MB.bin
File size: 10.00 MB
Total ngrams to process: 10,485,757
Progress: 0%..10%..20%..30%..40%..50%..60%..70%..80%..90%..100%
Processing complete!

Testing for first 4 bytes of file: 12345678
Ngram might be in the file

Testing for pattern: DEADBEEF
Ngram is definitely not in the file

Testing for pattern: CAFEBABE
Ngram is definitely not in the file

Testing for pattern: FFFFFFFF
Ngram is definitely not in the file
``` 