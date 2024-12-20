FROM python:3.10

# Set the working directory for the container
WORKDIR /app

# Copy only the necessary files from the protollm_tools/sdk directory
COPY protollm_tools/sdk .

# Install required dependencies
RUN pip install -r requirements.txt

# Default command to start Celery worker
CMD ["celery", "-A", "protollm_sdk.celery.app", "worker", "--loglevel=info"]
