# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies required for xgboost and other scientific libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire API directory into the container
# This assumes the Docker context is the /API folder itself
COPY . .

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Set environment variables
ENV MONGO_URI=mongodb://mongodb:27017
ENV DB_NAME=skillscapes
ENV PYTHONUNBUFFERED=1

# Command to run the application
# Since we are inside the API folder, we run main.py as a module relative to /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
