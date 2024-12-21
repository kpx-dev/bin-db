import os
import random
import string
from datetime import datetime

def generate_random_binary_file(file_path: str, size_mb: int = 10):
    """
    Generate a random binary file of specified size
    
    Args:
        file_path: Path where the binary file will be saved
        size_mb: Size of the file in megabytes
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert MB to bytes (1 MB = 1024 * 1024 bytes)
    size_bytes = size_mb * 1024 * 1024
    
    # Generate and write random bytes
    with open(file_path, 'wb') as f:
        f.write(os.urandom(size_bytes))

def generate_random_text_file(file_path: str, size_mb: int = 10):
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Calculate target size in bytes
    target_size = size_mb * 1024 * 1024
    current_size = 0
    
    # Create character set for random text (letters, numbers, punctuation)
    chars = string.ascii_letters + string.digits + string.punctuation + ' ' * 10  # extra spaces for readability
    
    print(f"Generating {size_mb}MB random text file...")
    
    with open(file_path, 'w') as f:
        # Write in chunks of 1KB for efficiency
        chunk_size = 1024
        
        while current_size < target_size:
            # Generate a random string chunk
            chunk = ''.join(random.choice(chars) for _ in range(chunk_size))
            chunk_bytes = chunk.encode('utf-8')
            
            # Check if adding this chunk would exceed target size
            if current_size + len(chunk_bytes) > target_size:
                remaining = target_size - current_size
                chunk = chunk[:remaining]
            
            f.write(chunk)
            current_size += len(chunk.encode('utf-8'))
            
            # Show progress
            progress = (current_size * 100) // target_size
            print(f"\rProgress: {progress}%", end='')
    
    # Get actual file size
    actual_size = os.path.getsize(file_path)
    print(f"\nGenerated random text file at: {file_path}")
    print(f"Actual file size: {actual_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate a 10MB random binary file in the data folder
    binary_path = os.path.join('data', f'random_binary_{timestamp}.bin')
    generate_random_binary_file(binary_path)
    
    # Verify binary file size
    binary_size_mb = os.path.getsize(binary_path) / (1024 * 1024)
    print(f"Generated random binary file at: {binary_path}")
    print(f"Binary file size: {binary_size_mb:.2f} MB")
    
    print("\n" + "="*50 + "\n")
    
    # Generate a 10MB random text file in the data folder
    text_path = os.path.join('data', f'random_text_{timestamp}.txt')
    generate_random_text_file(text_path) 