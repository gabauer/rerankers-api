# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml file to the working directory
COPY pyproject.toml .

# Install build dependencies required by setuptools
RUN pip install setuptools

# Install project dependencies specified in pyproject.toml
RUN pip install .

# Copy the rest of the application code
COPY . .

# Set environment variable to configure the ranker model (optional)
# ENV RANKER_MODEL=flashrank  # You can uncomment this line to set a default ranker model

# Expose port 8000 for the Uvicorn server
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "rest.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
