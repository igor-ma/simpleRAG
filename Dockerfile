FROM python:3.9

WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]