# Use official Python slim image
FROM python:3.12.9-slim-bullseye



# Install required tools
RUN apt-get update && apt-get install -y \
    curl gnupg2 ca-certificates apt-transport-https \
    unixodbc unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Add Microsoft package repository for Debian 11 (bullseye)
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg && \
    curl https://packages.microsoft.com/config/debian/11/prod.list \
        -o /etc/apt/sources.list.d/mssql-release.list

# Install SQL Server ODBC Driver 17
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17





# Set working directory
WORKDIR /app


# Install dotenv support (usually already installed)
RUN pip install python-dotenv



# Copy only requirements first (for caching)
COPY requirements.txt .


# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files (except ignored files)
COPY . .

# Expose Streamlit default port
EXPOSE 8502

# Run Streamlit
CMD ["streamlit", "run", "src/py_files/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
