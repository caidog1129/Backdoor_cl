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

# Read and print every line in the file
while IFS= read -r line; do
    echo "Running backdoor_search.py"
    nohup python backdoor_search/backdoor_search.py --instance_dir "$line" --config /nethome/aferber3/aaron/caijunya/backdoor_CL/backdoor_search/configs/mcts.ini --max_time 18000 --seed 0 &
    ((running_processes++))

    # Limit to 6 concurrent processes
    if [ "$running_processes" -eq 6 ]; then
        wait
        running_processes=0
    fi
    
done < "$1"

wait

while IFS= read -r line; do
    echo "Running backdoor_evaluate.py"
    nohup python backdoor_search/backdoor_evaluate.py --instance_dir "$line" &
done < "$1"

echo "All Python commands completed."

