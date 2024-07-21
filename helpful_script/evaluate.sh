#!/bin/bash

# Check if a filename is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "Error: File '$1' not found."
    exit 1
fi

while IFS= read -r line; do
    echo "Running evaluate.py"
    nohup python evaluate.py --instance_dir "$line" &
done < "$1"

echo "All Python commands completed."

