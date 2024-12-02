FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean
RUN git clone https://github.com/slokadjaja/thesis.git /app/code

WORKDIR /app/code

RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x /app/code/pipeline.sh

CMD ["/app/code/pipeline.sh"]
