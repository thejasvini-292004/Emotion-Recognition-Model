# Use an official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Dash runs on
EXPOSE 7860

# Run the app
CMD ["python", "app2.py"]
