#!/bin/bash

# Initialize the maximum memory usage variable
max_memory=0

# Function to get the current memory usage
get_memory_usage() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}'
}

# Monitor the memory usage
while true; do
    current_memory=$(get_memory_usage)
    if (( current_memory > max_memory )); then
        max_memory=$current_memory
        echo "Current memory usage: ${current_memory} MiB"
    fi
    sleep 0.1  # Adjust the sleep interval as needed
done

# Print the maximum memory usage
echo "Maximum memory used during training: ${max_memory} MiB"
