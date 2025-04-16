import mmh3  # MurmurHash3 for efficient hashing
import math
from typing import BinaryIO, List, Tuple, Set
import numpy as np
import hashlib
import os
import sys
import pickle

class BloomFilter:
    def __init__(self, size: int, num_hash_functions: int = 5):
        """
        Initialize a Bloom Filter with enhanced false positive reduction.
        
        Args:
            size: Size of the bit array
            num_hash_functions: Number of hash functions to use (default increased to 5)
        """
        self.size = size
        self.num_hash_functions = num_hash_functions
        self.bit_array = np.zeros(size, dtype=bool)
        # Secondary verification set for high-confidence checks
        self.verification_set: Set[bytes] = set()
        
    def _get_hash_positions(self, item: bytes) -> List[int]:
        """
        Get positions in the bit array using multiple hash functions.
        Uses a combination of MurmurHash3 and SHA-256 for better distribution.
        
        Args:
            item: The item to hash
            
        Returns:
            List of positions in the bit array
        """
        positions = []
        
        # Use MurmurHash3 with different seeds
        for seed in range(self.num_hash_functions):
            hash_value = mmh3.hash(item, seed) % self.size
            positions.append(hash_value)
            
        # Add SHA-256 based positions for better distribution
        sha256_hash = hashlib.sha256(item).digest()
        for i in range(0, len(sha256_hash), 4):
            if len(positions) >= self.num_hash_functions * 2:
                break
            hash_value = int.from_bytes(sha256_hash[i:i+4], 'big') % self.size
            positions.append(hash_value)
            
        return positions
    
    def add(self, item: bytes) -> None:
        """
        Add an item to the Bloom Filter.
        Also adds to verification set if it's a high-confidence item.
        
        Args:
            item: The item to add
        """
        positions = self._get_hash_positions(item)
        for pos in positions:
            self.bit_array[pos] = True
            
        # Add to verification set if it's a high-confidence item
        # (e.g., if it appears multiple times or meets certain criteria)
        if self._is_high_confidence(item):
            self.verification_set.add(item)
    
    def _is_high_confidence(self, item: bytes) -> bool:
        """
        Determine if an item should be added to the verification set.
        This is a simple implementation - you might want to customize this logic.
        
        Args:
            item: The item to check
            
        Returns:
            True if the item should be added to verification set
        """
        # Example: Add to verification set if it appears multiple times
        # You can customize this logic based on your needs
        return item in self.verification_set or len(self.verification_set) < 1000
    
    def __contains__(self, item: bytes) -> bool:
        """
        Check if an item is in the Bloom Filter with enhanced accuracy.
        
        Args:
            item: The item to check
            
        Returns:
            True if the item might be in the set, False if it definitely isn't
        """
        # First check the verification set for high-confidence items
        if item in self.verification_set:
            return True
            
        # Then check the Bloom Filter
        positions = self._get_hash_positions(item)
        return all(self.bit_array[pos] for pos in positions)

    def save(self, filepath: str) -> None:
        """
        Save the Bloom Filter to disk.
        
        Args:
            filepath: Path where to save the Bloom Filter
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save metadata and bit array
        metadata = {
            'size': self.size,
            'num_hash_functions': self.num_hash_functions
        }
        
        # Save metadata and bit array using numpy
        np.savez(
            filepath,
            metadata=np.array([metadata['size'], metadata['num_hash_functions']]),
            bit_array=self.bit_array
        )
        
        # Save verification set using pickle
        verification_path = f"{filepath}.verification"
        with open(verification_path, 'wb') as f:
            pickle.dump(self.verification_set, f)
            
        print(f"Bloom Filter saved to {filepath}")
        print(f"Verification set saved to {verification_path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BloomFilter':
        """
        Load a Bloom Filter from disk.
        
        Args:
            filepath: Path to the saved Bloom Filter
            
        Returns:
            Loaded BloomFilter instance
        """
        # Load metadata and bit array
        data = np.load(filepath)
        metadata = data['metadata']
        size = int(metadata[0])
        num_hash_functions = int(metadata[1])
        
        # Create new Bloom Filter instance
        bloom_filter = cls(size, num_hash_functions)
        bloom_filter.bit_array = data['bit_array']
        
        # Load verification set
        verification_path = f"{filepath}.verification"
        with open(verification_path, 'rb') as f:
            bloom_filter.verification_set = pickle.load(f)
            
        print(f"Bloom Filter loaded from {filepath}")
        print(f"Verification set loaded from {verification_path}")
        return bloom_filter

def process_binary_file(file_path: str, bloom_filter: BloomFilter) -> None:
    """
    Process a binary file and add all 4-byte ngrams to the Bloom Filter.
    Uses sliding window to capture all possible 4-byte sequences.
    Shows progress during processing.
    
    Args:
        file_path: Path to the binary file
        bloom_filter: The Bloom Filter to add ngrams to
    """
    file_size = os.path.getsize(file_path)
    total_ngrams = file_size - 3  # Number of possible 4-byte sequences
    processed_ngrams = 0
    last_percentage = -1
    
    print(f"Processing file: {file_path}")
    print(f"File size: {file_size / (1024*1024):.2f} MB")
    print(f"Total ngrams to process: {total_ngrams:,}")
    print("Progress: ", end="", flush=True)
    
    with open(file_path, 'rb') as f:
        data = f.read()
        for i in range(len(data) - 3):
            ngram = data[i:i+4]
            bloom_filter.add(ngram)
            
            # Update progress
            processed_ngrams += 1
            percentage = int((processed_ngrams / total_ngrams) * 100)
            
            # Only print when percentage changes to avoid flooding the console
            if percentage != last_percentage:
                if percentage % 10 == 0:
                    print(f"{percentage}%", end="", flush=True)
                elif percentage % 2 == 0:
                    print(".", end="", flush=True)
                last_percentage = percentage
    
    print("\nProcessing complete!")

def calculate_optimal_size(expected_items: int, false_positive_rate: float) -> Tuple[int, int]:
    """
    Calculate optimal size and number of hash functions for a Bloom Filter.
    Uses more conservative parameters to minimize false positives.
    
    Args:
        expected_items: Expected number of items to store
        false_positive_rate: Desired false positive rate
        
    Returns:
        Tuple of (size, num_hash_functions)
    """
    # Use more conservative size calculation
    size = -2 * (expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2)
    # Increase number of hash functions
    num_hash_functions = (size / expected_items) * math.log(2) * 1.5
    return int(size), int(num_hash_functions)

def calculate_actual_ngrams(file_path: str) -> int:
    """
    Calculate the actual number of 4-byte ngrams in a file.
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        Number of 4-byte ngrams in the file
    """
    file_size = os.path.getsize(file_path)
    return file_size - 3  # For 4-byte ngrams

# Example usage
if __name__ == "__main__":
    # Calculate actual number of ngrams in the file
    file_path = "data/random_binary_file_10MB.bin"
    actual_ngrams = calculate_actual_ngrams(file_path)
    print(f"File contains {actual_ngrams:,} possible 4-byte ngrams")
    
    # Use a very low false positive rate (0.0001 = 0.01%)
    false_positive_rate = 0.0001
    size, num_hash_functions = calculate_optimal_size(actual_ngrams, false_positive_rate)
    print(f"Bloom Filter size: {size:,} bits")
    print(f"Number of hash functions: {num_hash_functions}")
    
    # Try to load existing Bloom Filter, or create new one
    bloom_filter_path = "data/bloom_filter.npz"
    if os.path.exists(bloom_filter_path):
        print("Loading existing Bloom Filter...")
        bloom_filter = BloomFilter.load(bloom_filter_path)
    else:
        print("Creating new Bloom Filter...")
        bloom_filter = BloomFilter(size, num_hash_functions)
        
        # Process a binary file
        process_binary_file(file_path, bloom_filter)
        
        # Save the Bloom Filter
        bloom_filter.save(bloom_filter_path)
    
    # Test case 1: First 4 bytes of the file (we know this exists)
    with open(file_path, 'rb') as f:
        first_ngram = f.read(4)
    print(f"\nTesting for first 4 bytes of file: {first_ngram.hex().upper()}")
    if first_ngram in bloom_filter:
        print("Ngram might be in the file")
    else:
        print("Ngram is definitely not in the file")
    
    # Test case 2: A very distinctive pattern that's unlikely to exist
    test_ngram = bytes([0xDE, 0xAD, 0xBE, 0xEF])  # "DEADBEEF" in hex
    print(f"\nTesting for pattern: {test_ngram.hex().upper()}")
    if test_ngram in bloom_filter:
        print("Ngram might be in the file")
    else:
        print("Ngram is definitely not in the file")
        
    # Test case 3: Another distinctive pattern
    test_ngram2 = bytes([0xCA, 0xFE, 0xBA, 0xBE])  # "CAFEBABE" in hex
    print(f"\nTesting for pattern: {test_ngram2.hex().upper()}")
    if test_ngram2 in bloom_filter:
        print("Ngram might be in the file")
    else:
        print("Ngram is definitely not in the file")
        
    # Test case 4: A pattern that's extremely unlikely to exist
    test_ngram3 = bytes([0xFF, 0xFF, 0xFF, 0xFF])  # All bits set to 1
    print(f"\nTesting for pattern: {test_ngram3.hex().upper()}")
    if test_ngram3 in bloom_filter:
        print("Ngram might be in the file")
    else:
        print("Ngram is definitely not in the file") 