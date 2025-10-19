#!/usr/bin/env python3
"""
Entropy Coding Module for Integer Compression

This module provides functionality to compress lists of integers using entropy coding
with the constriction library. It determines the optimal bit size, encodes integers
to that bit size, and performs entropy compression.
"""

import numpy as np
import constriction
import struct
import math
from typing import List, Tuple, Optional
import os
import random


class IntegerEntropyEncoder:
    """
    A class for entropy encoding of integer sequences.
    
    This encoder:
    1. Analyzes the input integers to determine the maximum bit size needed
    2. Converts integers to the optimal bit representation
    3. Performs entropy encoding using adaptive arithmetic coding
    4. Outputs compressed binary data with metadata
    """
    
    def __init__(self):
        self.bit_size = None
        self.original_length = None
        
    def _calculate_optimal_bits(self, integers: List[int]) -> int:
        """
        Calculate the minimum number of bits needed to represent all integers.
        
        Args:
            integers: List of integers to analyze
            
        Returns:
            Number of bits needed to represent the maximum value
        """
        if not integers:
            return 1
            
        # Handle negative numbers by finding the range
        min_val = min(integers)
        max_val = max(integers)
        
        if min_val < 0:
            # For signed integers, we need to consider the full range
            range_val = max(abs(min_val), abs(max_val))
            # Add 1 bit for sign
            bits_needed = math.ceil(math.log2(range_val + 1)) + 1 if range_val > 0 else 2
        else:
            # For unsigned integers
            bits_needed = math.ceil(math.log2(max_val + 1)) if max_val > 0 else 1
            
        return max(1, bits_needed)
    
    def _normalize_integers(self, integers: List[int], bit_size: int) -> np.ndarray:
        """
        Normalize integers to fit within the specified bit size.
        
        Args:
            integers: Input integers
            bit_size: Target bit size
            
        Returns:
            Normalized integers as numpy array
        """
        integers = np.array(integers, dtype=np.int64)
        
        # If we have negative numbers, shift to make all positive
        min_val = integers.min()
        if min_val < 0:
            integers = integers - min_val
            
        # Ensure all values fit in the specified bit size
        max_allowed = (1 << bit_size) - 1
        integers = np.clip(integers, 0, max_allowed)
        
        return integers.astype(np.uint32)
    
    def _create_probability_model(self, symbols: np.ndarray, bit_size: int) -> np.ndarray:
        """
        Create a probability model for the symbols based on their frequency.
        
        Args:
            symbols: Array of symbols to encode
            bit_size: Bit size of the symbols
            
        Returns:
            Probability distribution array
        """
        alphabet_size = 1 << bit_size
        
        # Count frequencies
        counts = np.bincount(symbols, minlength=alphabet_size)
        
        # Add small epsilon to avoid zero probabilities
        epsilon = 1e-10
        probabilities = (counts + epsilon) / (len(symbols) + alphabet_size * epsilon)
        
        # Normalize to ensure sum is 1
        probabilities = probabilities / probabilities.sum()
        
        return probabilities
    
    def encode(self, integers: List[int], output_file: str) -> Tuple[int, int, float]:
        """
        Encode a list of integers and save to a binary file.
        
        Args:
            integers: List of integers to encode
            output_file: Path to output binary file
            
        Returns:
            Tuple of (bit_size_used, original_size_bits, compression_ratio)
        """
        if not integers:
            raise ValueError("Input list cannot be empty")
            
        # Step 1: Calculate optimal bit size
        self.bit_size = self._calculate_optimal_bits(integers)
        self.original_length = len(integers)
        
        # Step 2: Normalize integers to fit the bit size
        normalized_symbols = self._normalize_integers(integers, self.bit_size)
        
        # Step 3: Create probability model
        probabilities = self._create_probability_model(normalized_symbols, self.bit_size)
        
        # Step 4: Perform entropy encoding using arithmetic coding
        
        # Convert probabilities to the format expected by constriction
        alphabet_size = len(probabilities)
        compression_method = "FALLBACK"  # Default to fallback
        
        try:
            # Create an ANS encoder
            encoder = constriction.stream.stack.AnsCoder()
            
            # Create entropy model using categorical distribution with proper numpy array
            probabilities_float32 = probabilities.astype(np.float32)
            
            # Create categorical entropy model (constriction handles normalization)
            entropy_model = constriction.stream.model.Categorical(probabilities_float32, perfect=False)
            
            # Convert symbols to int32 numpy array (required by constriction)
            symbols_int32 = normalized_symbols.astype(np.int32)
            
            # Encode all symbols at once using encode_reverse
            encoder.encode_reverse(symbols_int32, entropy_model)
                
            # Get compressed data as numpy array
            compressed_array = encoder.get_compressed()
            
            # Convert numpy array to bytes
            compressed_data = compressed_array.tobytes()
            
            # Store the probability model and method for decoding
            self.compression_method = "ANS"
            self.probabilities = probabilities_float32
            self.compressed_array = compressed_array  # Store for decoding
            compression_method = "ANS"
            
        except Exception as e:
            # Fallback: use a simpler approach
            compressed_data = self._fallback_encode(normalized_symbols)
            self.compression_method = "FALLBACK"
            compression_method = "FALLBACK"

        # Step 5: Write to binary file with header
        # Get probabilities for ANS method
        probs_to_store = None
        if compression_method == "ANS" and hasattr(self, 'probabilities'):
            probs_to_store = self.probabilities
            
        self._write_compressed_file(output_file, compressed_data, integers, compression_method, probs_to_store)        # Calculate compression statistics
        original_size_bits = len(integers) * self.bit_size
        # Header size: 6 fields * 4 bytes + probability table size
        header_bits = 6 * 4 * 8
        if compression_method == "ANS" and hasattr(self, 'probabilities'):
            header_bits += len(self.probabilities) * 4 * 8  # float32 probabilities
        
        compressed_size_bits = len(compressed_data) * 8 + header_bits
        compression_ratio = original_size_bits / compressed_size_bits if compressed_size_bits > 0 else 0
        
        return self.bit_size, original_size_bits, compression_ratio, compression_method, len(compressed_data), len(integers)
    
    def _fallback_encode(self, symbols: np.ndarray) -> bytes:
        """
        Fallback encoding method using simple bit packing.
        """
        # Pack symbols into bytes
        packed_data = b''
        bit_buffer = 0
        bits_in_buffer = 0
        
        for symbol in symbols:
            bit_buffer = (bit_buffer << self.bit_size) | int(symbol)
            bits_in_buffer += self.bit_size
            
            while bits_in_buffer >= 8:
                byte_val = (bit_buffer >> (bits_in_buffer - 8)) & 0xFF
                packed_data += bytes([byte_val])
                bits_in_buffer -= 8
        
        # Handle remaining bits
        if bits_in_buffer > 0:
            byte_val = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
            packed_data += bytes([byte_val])
            
        return packed_data
    
    def _write_compressed_file(self, output_file: str, compressed_data: bytes, original_integers: List[int], 
                              compression_method: str = "ANS", probabilities: np.ndarray = None):
        """
        Write compressed data to file with header information.
        
        File format:
        - 4 bytes: bit size used for encoding
        - 4 bytes: original number of integers
        - 4 bytes: size of compressed data
        - 4 bytes: minimum value (for negative number handling)
        - 4 bytes: compression method (0=fallback, 1=ANS)
        - 4 bytes: alphabet size (for ANS decoding)
        - N*4 bytes: probability table (for ANS, as float32 values)
        - N bytes: compressed data
        """
        min_val = min(original_integers)
        method_code = 1 if compression_method == "ANS" else 0
        
        with open(output_file, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', self.bit_size))  # 4 bytes: bit size
            f.write(struct.pack('<I', self.original_length))  # 4 bytes: original length
            f.write(struct.pack('<I', len(compressed_data)))  # 4 bytes: compressed size
            f.write(struct.pack('<i', min_val))  # 4 bytes: minimum value (signed)
            f.write(struct.pack('<I', method_code))  # 4 bytes: compression method
            
            # Write probability table for ANS
            if compression_method == "ANS" and probabilities is not None:
                alphabet_size = len(probabilities)
                f.write(struct.pack('<I', alphabet_size))  # 4 bytes: alphabet size
                # Write probabilities as float32
                for prob in probabilities:
                    f.write(struct.pack('<f', prob))
            else:
                f.write(struct.pack('<I', 0))  # No probabilities for fallback method
            
            # Write compressed data
            f.write(compressed_data)
    
    def decode(self, input_file: str) -> List[int]:
        """
        Decode integers from a compressed binary file.
        
        Args:
            input_file: Path to compressed binary file
            
        Returns:
            List of decoded integers
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        with open(input_file, 'rb') as f:
            # Read header
            bit_size = struct.unpack('<I', f.read(4))[0]
            original_length = struct.unpack('<I', f.read(4))[0]
            compressed_size = struct.unpack('<I', f.read(4))[0]
            min_val = struct.unpack('<i', f.read(4))[0]
            method_code = struct.unpack('<I', f.read(4))[0]
            alphabet_size = struct.unpack('<I', f.read(4))[0]
            
            # Read probability table for ANS
            probabilities = None
            if method_code == 1 and alphabet_size > 0:  # ANS method
                probabilities = np.zeros(alphabet_size, dtype=np.float32)
                for i in range(alphabet_size):
                    probabilities[i] = struct.unpack('<f', f.read(4))[0]
            
            # Read compressed data
            compressed_data = f.read(compressed_size)
        
        compression_method = "ANS" if method_code == 1 else "FALLBACK"
        
        if compression_method == "ANS" and probabilities is not None:
            try:
                decoded_symbols = self._ans_decode(compressed_data, bit_size, original_length, probabilities)
            except Exception as e:
                decoded_symbols = self._fallback_decode(compressed_data, bit_size, original_length)
        else:
            decoded_symbols = self._fallback_decode(compressed_data, bit_size, original_length)
        
        # Restore original range
        if min_val < 0:
            decoded_symbols = [x + min_val for x in decoded_symbols]
        
        return decoded_symbols
    
    def _ans_decode(self, compressed_data: bytes, bit_size: int, length: int, probabilities: np.ndarray) -> List[int]:
        """
        Decode using ANS decoder with the stored probability model.
        
        Args:
            compressed_data: The compressed byte data
            bit_size: Bit size used during encoding  
            length: Number of symbols to decode
            probabilities: Probability distribution used during encoding
            
        Returns:
            List of decoded symbols
        """
        # Convert bytes back to numpy array (reverse of tobytes())
        compressed_array = np.frombuffer(compressed_data, dtype=np.uint32)
        
        # Create ANS decoder
        decoder = constriction.stream.stack.AnsCoder(compressed_array)
        
        # Recreate the same entropy model used during encoding
        entropy_model = constriction.stream.model.Categorical(probabilities, perfect=False)
        
        # Decode symbols
        decoded_symbols = decoder.decode(entropy_model, length)
        
        return decoded_symbols.tolist()
    
    def _fallback_decode(self, data: bytes, bit_size: int, length: int) -> List[int]:
        """
        Fallback decoder for bit-packed data.
        """
        symbols = []
        bit_buffer = 0
        bits_in_buffer = 0
        
        for byte in data:
            bit_buffer = (bit_buffer << 8) | byte
            bits_in_buffer += 8
            
            while bits_in_buffer >= bit_size and len(symbols) < length:
                symbol = (bit_buffer >> (bits_in_buffer - bit_size)) & ((1 << bit_size) - 1)
                symbols.append(symbol)
                bits_in_buffer -= bit_size
                
                if len(symbols) >= length:
                    break
        
        return symbols[:length]


def compress_integers(integers: List[int], output_file: str) -> dict:
    """
    Convenience function to compress a list of integers.
    
    Args:
        integers: List of integers to compress
        output_file: Path to output compressed file
        
    Returns:
        Dictionary with compression statistics
    """
    encoder = IntegerEntropyEncoder()
    bit_size, original_bits, compression_ratio, compression_method, compressed_size, num_symbols = encoder.encode(integers, output_file)
    
    return {
        'bit_size': bit_size,
        'original_size_bits': original_bits,
        'compressed_file': output_file,
        'compression_ratio': compression_ratio,
        'original_count': len(integers),
        'compression_method': compression_method,
        'compressed_size': compressed_size,
        'num_symbols': num_symbols
    }


def decompress_integers(input_file: str) -> List[int]:
    """
    Convenience function to decompress integers from a file.
    
    Args:
        input_file: Path to compressed file
        
    Returns:
        List of decompressed integers
    """
    encoder = IntegerEntropyEncoder()
    return encoder.decode(input_file)


if __name__ == "__main__":
    # Example usage
    print("=== Integer Entropy Coding Example ===")
    
    # Generate some test data
    num_test_integers = 128
    max_value = 2**13  # Adjusted to match bit size of 13
    test_integers = [random.randint(1, max_value) for _ in range(num_test_integers)]
    print(f"Original integers: {test_integers[:20]}... (total: {len(test_integers)})")
    
    # Compress
    output_file = "compressed_integers.bin"
    print(f"\n=== Compression Process ===")
    print(f"Starting entropy encoding...")
    
    stats = compress_integers(test_integers, output_file)
    
    print(f"Optimal bit size determined: {stats['bit_size']} bits")
    print(f"Number of symbols: {stats['num_symbols']}")
    
    if stats['compression_method'] == 'ANS':
        print(f"Successfully encoded using ANS coding")
        print(f"Compressed to {stats['compressed_size']} bytes from {stats['num_symbols']} symbols")
    else:
        print("Using fallback encoding method...")
        print("Falling back to bit packing compression...")
    
    print(f"Compressed data written to: {stats['compressed_file']}")
    print(f"Header: {stats['bit_size']} bits per symbol, {stats['original_count']} symbols")
    print(f"Compression method: {stats['compression_method']}")
    print(f"Compressed size: {stats['compressed_size']} bytes")
    
    print(f"\n=== Compression Results ===")
    print(f"Bit size used: {stats['bit_size']} bits")
    print(f"Original size: {stats['original_size_bits']} bits")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Output file: {stats['compressed_file']}")
    
    # Decompress and verify
    print(f"\n=== Decompression Process ===")
    print(f"Performing ANS decoding..." if stats['compression_method'] == 'ANS' else "Performing fallback decoding...")
    
    decoded = decompress_integers(output_file)
    
    print(f"Decoding: {stats['bit_size']} bits per symbol, {stats['original_count']} symbols")
    print(f"Compression method: {stats['compression_method']}")
    
    if stats['compression_method'] == 'ANS':
        print(f"Successfully decoded {len(decoded)} symbols using ANS")
    
    print(f"Decoded integers: {decoded[:20]}... (total: {len(decoded)})")
    
    # Verify accuracy
    if decoded == test_integers:
        print("✓ Compression and decompression successful!")
    else:
        print("✗ Decompression mismatch detected")
        print(f"Original length: {len(test_integers)}, Decoded length: {len(decoded)}")