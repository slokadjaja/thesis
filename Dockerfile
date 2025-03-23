FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Combine all apt-get commands into one RUN to reduce layers
RUN apt-get update && \
    apt-get install -y git wget unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone your repository
RUN git clone https://github.com/slokadjaja/thesis.git /app/code

WORKDIR /app/code

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make sure the pipeline.sh script is executable
RUN chmod +x /app/code/pipeline.sh

# Download and unzip the file
RUN wget -P /app/code/benchmarks/VQShape/ https://github.com/YunshiWen/VQShape/releases/download/v0.1.0-cls/uea_dim256_codebook512.zip && \
    unzip /app/code/benchmarks/VQShape/uea_dim256_codebook512.zip -d /app/code/benchmarks/VQShape/

# Copy the .env file
COPY .env /app/code/.env

# Set the default command to bash
CMD ["bash"]
