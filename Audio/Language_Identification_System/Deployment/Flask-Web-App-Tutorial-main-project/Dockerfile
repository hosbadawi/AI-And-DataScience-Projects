#Use python as base image
FROM python:3.7.9-slim

#Use working directory /app
WORKDIR /app

#Copy all the content of current directory to /app
ADD . /app

#Installing required packages
RUN python -m pip install --upgrade pip

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 

COPY requirements.txt requirements.txt

RUN pip install --trusted-host pypi.python.org -r requirements.txt
  
#Open port 3000
EXPOSE 5000

#Set environment variable
ENV NAME OpentoAll

#Run python program
CMD ["python","main.py"]


 