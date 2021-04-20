FROM tensorflow/tensorflow:2.4.1
RUN mkdir /DeepFeatSelection
COPY . /DeepFeatSelection/
RUN pip install pandas==1.1.5

