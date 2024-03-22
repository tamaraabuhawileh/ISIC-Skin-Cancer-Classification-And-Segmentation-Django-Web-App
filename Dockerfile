FROM debian:11

RUN apt update
RUN apt -fy install python3.9
RUN apt -fy install python3-pip

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TF_ENABLE_ONEDNN_OPTS 0

WORKDIR /app

COPY ./requirements.txt /app/

RUN apt-get update
RUN apt-get -y install libglib2.0-0
RUN apt-get -y install libsm6 \
     libxrender-dev \
     libxext6 \
     && apt-get -y install libpq-dev gcc \
     && pip install psycopg2
RUN pip install opencv-python-headless==4.5.3.56

# Install virtualenv
RUN pip install -r requirements.txt
RUN pip install virtualenv

# Create a virtual environment named newenv
RUN virtualenv newenv

# Activate the virtual environment
RUN /bin/bash -c "source newenv/bin/activate"

# Install Django within the virtual environment
RUN pip install django
RUN pip install pandas
COPY . /app

EXPOSE 8000

CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
