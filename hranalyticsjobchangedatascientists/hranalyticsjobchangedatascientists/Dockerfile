# This sets the base image for the Docker container to be the official Python 3.11 image from Docker
FROM python:3.11

# This sets the working directory inside the container to /sc. This is where the application code wi*
WORKDIR /HRAnalyticsJobChangeDataScientists/hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists

# This copies the requirements.txt file from the host (the directory where the Dockerfile is located)
COPY requirements-310.txt ./

# This command runs inside the container and installs the Python packages listed in the requirements.
RUN pip3 install --no-cache-dir -r requirements-310.txt

# This copies the entire sc directory from the host to the /src directory inside the container. The
COPY . ./

# This command updates the package index inside the container and then installs the vim text editor.
RUN apt-get update && apt-get install -y vim

# This instruction indicates that the container will expose port 8000. However, it doesn't actually pi
EXPOSE 8000

# This sets the default command that will be executed when the container starts. It runs the Uvicorn I
CMD ["uvicorn", "main:app","--host", "0.0.0.0","--port", "8000","--reload"]