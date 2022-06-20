# our base image
FROM tensorflow/tensorflow:latest-gpu

# Install python and pip
RUN apt install python3-pip

# upgrade pip
RUN pip install --upgrade pip
RUN apt install -y graphviz


# copy files required for the app to run
COPY . .

# install Python modules needed by the Python app
RUN pip install --no-cache-dir -r requirements.txt

# tell the port number the container should expose
EXPOSE 5000

# run the application
CMD ["python3", "-m", "src.models.train_model"]
