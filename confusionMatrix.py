import GetALAimages
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateBatch, ImageUrlCreateEntry, ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import GetALAimages
import csv
import time
import numpy as np
from PIL import Image
import urllib
from io import BytesIO
import math
from predictFromImageUrl import predictFromImageUrl


def predict(image_url, iteration_number):
    # get prediction percentages
    try:
        results = predictor.classify_image_url(project.id, iteration_number, image_url)

    # not sure what exception should be as custom vision gives a strange error.
    except:
        print('resizing')
        try:
            # if images are too large for prediction (must be <4mb), use python to scale down
            img = urllib.request.urlopen(image_url)
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

            results = predictor.classify_image(project.id, iteration_number, imgByteArr)

        # if image actually doesn't exist
        except OSError:
            pass

    try:
        for tag in results.predictions:
            if tag.tag_name == 'cane toad':
                percentage = tag.probability
    # if image doesn't exist, results are not defined
    except UnboundLocalError:
        percentage = 'NA'

    return percentage




# setting up project using keys

ENDPOINT = "https://canetoadmachinelearning.cognitiveservices.azure.com/"

training_key = "cde7deba2d5d4df5b768b50b700c46b7"
prediction_key = "fb49a542a16a47e6b68b2983db158c32"
prediction_resource_id = "/subscriptions/baa59b08-5ec4-44ea-a907-b12782d8e2a0/resourceGroups/Canetoads/providers/Microsoft.CognitiveServices/accounts/CaneToadMachineLea-Prediction"


credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# finding project id
for project in trainer.get_projects():
    if project.name == 'iterative':
        break
publish_iteration_name = "Iteration1.8"

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)




# split labelled data into tuples of training and testing (C for cane toad, N for not)
CtrainingURLS = []
CtestingURLS = []
NtrainingURLS = []
NtestingURLS = []
# getting image urls using ALA file for each species (iterative as we can only upload 64 images at a time)
for species in ['caneToad', 'stripedMarshFrog', 'ornateBurrowingFrog', 'australianGreenTreeFrog', 'bumpyRockFrog',
                'crawlingToadlet', 'daintyGreenTreeFrog', 'desertFroglet', 'desertTreeFrog', 'giantFrog', 'hootingFrog',
                'longFootedFrog', 'marbledFrog', 'moaningFrog', 'motorbikeFrog', 'newHollandFrog', 'rockholeFrog',
                'rothsTreeFrog', 'westernBanjoFrog', 'whiteLippedTreeFrog']:

    if species=='caneToad':
        tag_name = species
    else:
        tag_name = 'otherFrog'

    file_dir = 'ala image urls/' + species + 'RawFile.csv'
    image_url_list = GetALAimages.listOfAlaImageUrls(file_dir)
    if species=='caneToad':
        CtestingURLS += image_url_list
    else:
        NtestingURLS += image_url_list


# build a control for testing afterwards
CcontrolURLS = []
NcontrolURLS = []

for i in range(0,50)*np.floor(len(CtestingURLS)/51):
    CcontrolURLS.append(CtestingURLS[int(i)])
    CtestingURLS.remove(CtestingURLS[int(i)])
for i in range(0,50)*np.floor(len(NtestingURLS)/51):
    NcontrolURLS.append(NtestingURLS[int(i)])
    NtestingURLS.remove(NtestingURLS[int(i)])


# create CSV file
test_urls = []
for url in CcontrolURLS:
    test_urls.append([url, 'ala cane toad', 'na','na','na'])
for url in NcontrolURLS:
    test_urls.append([url, 'ala other frog', 'na','na','na'])
predictFromImageUrl(test_urls, 'all_controlled')


# making confusion matrix
Ccorrect = 0
Ncorrect = 0
for control in CcontrolURLS:
    try:
        if predict(control, 'Iteration1.10')>0.90:
            Ccorrect+=1
    except TypeError:
        CcontrolURLS.remove(control)

for control in NcontrolURLS:
    try:
        if predict(control, 'Iteration1.10')<0.90:
            Ncorrect+=1
    except TypeError:
        NcontrolURLS.remove(control)

print(len(CcontrolURLS))
print(Ccorrect)
print(len(NcontrolURLS))
print(Ncorrect)

