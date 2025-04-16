import base64
import hashlib

def read_text_file(file_path):
    """Read the entire text file and return its contents."""
    with open(file_path, 'r') as file:
        return file.read()
    
def read_binary_file(file_path):
    """Read the entire binary file and return its contents."""
    with open(file_path, 'rb') as file:
        return file.read()

def save_base64_encoded(data, output_path):
    """Convert data to Base64 and save it to a file."""
    base64_encoded = base64.b64encode(data).decode('utf-8')
    with open(output_path, 'w') as file:
        file.write(base64_encoded)

def save_hex_encoded(data, output_path):
    """Convert data to hexadecimal and save it to a file."""
    hex_encoded = data.hex()
    with open(output_path, 'w') as file:
        file.write(hex_encoded)

def display_sha256_hash(data):
    """Compute and display the SHA-256 hash of the binary data."""
    sha256_hash = hashlib.sha256(data).hexdigest()
    print(f"\nSHA-256 hash of the binary file: {sha256_hash}")

def output_first_100_bytes(data):
    """Output the first 100 bytes in binary, Base64, and hexadecimal formats."""
    first_100_bytes = data[:100]
    
    # Binary representation
    binary_representation = ' '.join(format(byte, '08b') for byte in first_100_bytes)
    
    # Base64 representation
    base64_representation = base64.b64encode(first_100_bytes).decode('utf-8')
    
    # Hexadecimal representation
    hex_representation = first_100_bytes.hex()
    
    print("First 100 bytes in binary:")
    print(binary_representation)
    print("\nFirst 100 bytes in Base64:")
    print(base64_representation)
    print("\nFirst 100 bytes in hexadecimal:")
    print(hex_representation)

def main():
    # input_file_path = 'data/random_binary_file.bin'
    input_file_path = 'data/simple_text.txt'
    # base64_output_path = 'data/random_binary_file_base64.txt'
    # hex_output_path = 'data/random_binary_file_hex.txt'
    
    binary_data = read_binary_file(input_file_path)
    binary_data = read_text_file(input_file_path)
    
    # Save Base64 and hex encoded files
    # save_base64_encoded(binary_data, base64_output_path)
    # save_hex_encoded(binary_data, hex_output_path)
    
    # Display the SHA-256 hash
    # display_sha256_hash(binary_data)

    # Output the first 100 bytes
    # output_first_100_bytes(binary_data)

if __name__ == "__main__":
    main()