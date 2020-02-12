FROM python:3.6

ADD app.py /
ADD CC.csv /

RUN pip install flask
RUN pip install pandas
RUN pip install numpy 
RUN pip install sklearn
run pip install boto3

RUN mkdir “templates”
ADD templates/home.html /templates/
ADD templates/result.html /templates/

EXPOSE 5000
CMD python app.py