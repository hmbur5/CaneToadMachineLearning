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

    if results is not None:
        for tag in results.predictions:
            if tag.tag_name == 'cane toad':
                percentage = tag.probability
    else:
        percentage = 'NA'

    return percentage




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
# iteration name must be changed each iteration to publish
publish_iteration_name = "1"

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

    if species=='caneToad' or species =='questionCaneToad':
        tag_name = species
    else:
        tag_name = 'otherFrog'


    if species =='caneToad':
        image_url_list = GetALAimages.listOfCheckedImages('ala image urls/confirmedCaneToads.csv')
        CtestingURLS = image_url_list
    elif species=='questionableCaneToad':
        image_url_list = GetALAimages.listOfCheckedImages('ala image urls/confirmedNotCaneToads.csv')
    else:
        file_dir = 'ala image urls/' + species + 'RawFile.csv'
        image_url_list = GetALAimages.listOfAlaImageUrls(file_dir)
        NtestingURLS += image_url_list


# build a control for testing afterwards
CcontrolURLS = []
NcontrolURLS = []

for i in range(0,50)*np.floor(len(CtestingURLS)/50):
    CcontrolURLS.append(CtestingURLS[int(i)])
    CtestingURLS.remove(CtestingURLS[int(i)])
for i in range(0,50)*np.floor(len(NtestingURLS)/50):
    NcontrolURLS.append(NtestingURLS[int(i)])
    NtestingURLS.remove(NtestingURLS[int(i)])


# create CSV file
test_urls = []
for url in CcontrolURLS:
    test_urls.append([url, 'ala cane toad'])
for url in NcontrolURLS:
    test_urls.append([url, 'ala other frog'])
predictFromImageUrl(test_urls, 'all_controlled')


# making confusion matrix
Ccorrect = 0
Ncorrect = 0
for control in CcontrolURLS:
    #print(predict(control, 'Iteration15'))
    if predict(control, 'Iteration1')>0.90:
        Ccorrect+=1
for control in NcontrolURLS:
    #print(predict(control, 'Iteration15'))
    if predict(control, 'Iteration1')<0.90:
        Ncorrect+=1

print(len(CcontrolURLS))
print(Ccorrect)
print(len(NcontrolURLS))
print(Ncorrect)

