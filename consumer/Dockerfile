FROM python:3.6

WORKDIR /usr/

ADD ./requirements.txt ./
RUN pip install -r requirements.txt
ADD ./ ./

CMD ["python","consumer.py"]