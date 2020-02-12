FROM python:3.6

ADD model_to_s3.py /
ADD CC.csv /

RUN pip install boto
RUN pip install pandas
RUN pip install numpy 
RUN pip install sklearn

CMD [ "python", "./model_to_s3.py" ]

ENTRYPOINT  ["python","./model_to_s3.py"]

CMD myinput