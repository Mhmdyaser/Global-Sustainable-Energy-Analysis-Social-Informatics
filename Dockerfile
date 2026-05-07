
FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    libnss3 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .


CMD ["python", "code.py"]