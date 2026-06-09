#!/usr/bin/env bash

SOURCE_DIR="notebooks"
OUTPUT_DIR="converted_md"
OUTPUT_FORMAT="markdown"

# Exit immediately if a command exits with a non-zero status.
set -e

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory '$SOURCE_DIR' not found."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob # Ensures that the loop doesn't run if no files match
notebooks=("$SOURCE_DIR"/*.ipynb)

# Check if any notebook files were found
if [ ${#notebooks[@]} -eq 0 ]; then
    echo "No .ipynb files found in '$SOURCE_DIR'."
    exit 0
fi

echo "Found ${#notebooks[@]} notebooks to convert to $OUTPUT_FORMAT."

# 4. Loop through each notebook and convert it
for notebook in "${notebooks[@]}"; do
  # Get the base name of the file for a cleaner message
  filename=$(basename "$notebook")
  echo "--> Processing '$filename'..."

  # Run the nbconvert command
  jupyter nbconvert \
    --to "$OUTPUT_FORMAT" \
    --output-dir="$OUTPUT_DIR" \
    "$notebook"
done

echo "" # Newline for better formatting
echo "✅ All notebooks have been successfully converted to '$OUTPUT_DIR'."
