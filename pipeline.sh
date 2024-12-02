#!/bin/bash

# Check if the user is already logged in
LOGIN_STATUS=$(huggingface-cli whoami 2>&1)

# If not logged in, `whoami` will return a message about not being authenticated
if [[ "$LOGIN_STATUS" == *"Not logged in"* ]]; then
    echo "You are not logged in to Hugging Face CLI."
    echo "Please log in now."
    huggingface-cli login
else
    echo "You are logged in as:"
    echo "$LOGIN_STATUS"
fi

echo "Running the pipeline script..."
python pipeline.py