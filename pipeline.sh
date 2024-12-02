#!/bin/bash

echo "Please log in to Hugging Face CLI..."
huggingface-cli login

echo "Logged in. Running the pipeline script..."
python pipeline.py