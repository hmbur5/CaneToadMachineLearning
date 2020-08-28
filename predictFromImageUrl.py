from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import csv


def predict(image_url):

    ENDPOINT = "https://canetoadmachinelearning.cognitiveservices.azure.com/"

    training_key = "cde7deba2d5d4df5b768b50b700c46b7"
    prediction_key = "fb49a542a16a47e6b68b2983db158c32"
    prediction_resource_id = "/subscriptions/baa59b08-5ec4-44ea-a907-b12782d8e2a0/resourceGroups/Canetoads/providers/Microsoft.CognitiveServices/accounts/CaneToadMachineLea-Prediction"


    credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
    trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

    project = trainer.get_project('c3656572-45da-465e-9614-3eb4a5041625')
    publish_iteration_name = "classifyModel_urls"

    # Now there is a trained endpoint that can be used to make a prediction
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)



    results = predictor.classify_image_url(
        project.id, publish_iteration_name, image_url)

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))



# open urls for testing
with open('predictions/binaryAll.csv', 'wb') as myfile:
    for url in myfile:
        print(url)

        # if in right form predict url
        predict(url)

        # figure out how to add percentage to csv

