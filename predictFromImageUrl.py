from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import csv
from PIL import Image
import urllib
from io import BytesIO
import math


import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


# setting up project using keys
ENDPOINT = "https://canetoad-prediction.cognitiveservices.azure.com/"

# using hmbur5@student.monash.edu
training_key = "567bc1a3b9d4479283887d68c1d7f46c"
prediction_key = "6b9d10b6c8bc42878d92fe94213256a1"
prediction_resource_id = "/subscriptions/d0bdc746-b59f-4a0c-b651-9865282bcd1a/resourceGroups/CaneToadMachineLearning/providers/Microsoft.CognitiveServices/accounts/CaneToad-Prediction"


credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# finding project id
for project in trainer.get_projects():
    if project.name == 'all':
        break
publish_iteration_name = "Iteration1"

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)




def predictFromImageUrl(testing_image_urls, file_name):
    '''
    Takes a list of image urls, and saves a csv and html file giving the probability of each photo being a cane toad
    :param testing_image_urls: a list of image urls, where each element in the list is a list of two elements; the url
    and the label/description of the image
    :param file_name: directory for csv and html file to be placed in.
    :return:
    '''

    # get prediction percentages
    image_predictions = []
    for url,species in testing_image_urls:
        try:
            results = predictor.classify_image_url(project.id, publish_iteration_name, url)

        # not sure what exception should be as custom vision gives a strange error.
        except:
            print('too large')
            try:
                # if images are too large for prediction (must be <4mb), use python to scale down
                img = urllib.request.urlopen(url)
                img = Image.open(img)

                # resize
                height,width = img.size
                currentSize = len(img.fp.read())
                # these numbers could probably be better
                maxDim = math.floor(max([width,height]) * 4194304 / currentSize / 3)
                img.thumbnail((maxDim, maxDim))


                # convert back to byte object for classify_image
                imgByteArr = BytesIO()
                img.save(imgByteArr, format='PNG')
                imgByteArr = imgByteArr.getvalue()

                results = predictor.classify_image(project.id, publish_iteration_name, imgByteArr)

            # if image actually doesn't exist
            except OSError:
                pass

        if results is not None:
            for tag in results.predictions:
                if tag.tag_name == 'cane toad':
                    percentage = tag.probability
        else:
            percentage = 'NA'

        print(percentage)
        image_predictions.append([url, species, percentage])

    # get in ascending order of probability
    sorted_predictions = sorted(image_predictions, key=lambda tup: tup[2])


    # write new file with image urls and prediction percentages
    with open('predictions/'+file_name+'.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter = ',')
        wr.writerows(sorted_predictions)

    # write html file
    with open('predictions/'+file_name+'.html', 'w') as myfile:
        myfile.write('<!doctype html> <html> <head> <meta charset="UTF-8"> <title>Untitled Document</title> </head>  <body><table>')
        myfile.write('<tr><th>Image</th><th>Cane toad prob</th></tr>')
        for url, species, percentage in sorted_predictions:
            myfile.write('<tr>')
            myfile.write("<td><img src='"+ url + "' width='250' alt=''/></td>")
            myfile.write("<td>"+species+"</td>")
            myfile.write("<td>"+str(percentage)+"</td>")
            myfile.write('</tr>')





# to use for photos that werent in training
def predict_from_csv():
    # initialise list to store image urls from file
    testing_image_urls = []
    with open('predictions/binaryAll.csv', 'r') as myfile:
        for url in myfile:
            url, species = url.split(',')
            testing_image_urls.append([url, species])