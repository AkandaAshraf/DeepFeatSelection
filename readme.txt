Read how_it_works.pdf to know how it works!

The best way to run is to use build the docker image and run it from there.
Download this directory. Make sure docker and docker-compose is installed in the system. The  follow these steps:
1.	cd to the downloaded directory
2.	docker-compose build
3.	If you wish to use the dataset given in the task then use the default env variables in .env. nothing needs to be changed. However, any dataset can be used, just specify them in the env var parameters, with the attribute names as a csv file. See files in the volume for the data format.
- Data:/DeepFeatSelection/Data
- Models:/DeepFeatSelection/Models
- ExpOutput:/DeepFeatSelection/ExpOutput
4.	docker-compose run experiment
5.	provide the number of iterations/ models you wish to train
6.	the results will be generated in the specified file, e.g. EXP_OUTPUT_CSV_FILE=/DeepFeatSelection/ExpOutput/output_weights.csv [not feature-wise averaged]
7.	The ranked features will also be provided in the console with the weights/importance of each of the features
email: akandaashraf@outlook.com / Akanda Ashraf
