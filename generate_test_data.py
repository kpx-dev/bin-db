import os
import random
from datetime import datetime
from lorem_text import lorem

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
    """Generate a random text file with Lorem Ipsum content"""
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Calculate target size in bytes
    target_size = size_mb * 1024 * 1024
    current_size = 0
    
    print(f"Generating {size_mb}MB random text file...")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        while current_size < target_size:
            # Generate a paragraph of Lorem Ipsum
            paragraph = lorem.paragraph()
            
            # Add two newlines after each paragraph
            content = paragraph + '\n'
            content_bytes = content.encode('utf-8')
            
            # Check if adding this content would exceed target size
            if current_size + len(content_bytes) > target_size:
                remaining = target_size - current_size
                content = content[:remaining]
            
            f.write(content)
            current_size += len(content_bytes)
            
            # Show progress
            progress = (current_size * 100) // target_size
            print(f"\rProgress: {progress}%", end='')
    
    # Get actual file size
    actual_size = os.path.getsize(file_path)
    print(f"\nGenerated random text file at: {file_path}")
    print(f"File size: {actual_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate a random text file in the data folder
    text_path = os.path.join('data', f'random_text_{timestamp}.txt')
    generate_random_text_file(text_path, 1) 

    text_size_mb = os.path.getsize(text_path) / (1024 * 1024)
    print(f"Text file size: {text_size_mb:.2f} MB")
    
    print("\n" + "="*50 + "\n")