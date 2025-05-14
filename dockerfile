# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy app code
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --default-timeout=120 --retries=10 -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["python", "-m", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=localhost"]
