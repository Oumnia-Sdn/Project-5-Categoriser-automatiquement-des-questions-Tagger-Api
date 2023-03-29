FROM python:3.10.8-buster
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install python-multipart

CMD uvicorn fast:app --reload --port $PORT
