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
from preCrop import cropImage
import numpy as np


import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


# setting up project using keys
ENDPOINT = "https://canetoadmodel-prediction.cognitiveservices.azure.com/"

# using nic
training_key = "af562773faaa490eb5028a14ded3b8cc"
prediction_key = "8043e5cca5634caaab92697bf568942d"
prediction_resource_id = "/subscriptions/79ac0136-fad4-4fe7-bda8-aec4a67de458/resourceGroups/CaneToads/providers/Microsoft.CognitiveServices/accounts/CaneToadModel-Prediction"


credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# finding project id
for project in trainer.get_projects():
    if project.name == 'all':
        break
publish_iteration_name = "Iteration6"


# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)



def predictFromImageFile(open_image):
    # convert back to byte object for classify_image
    imgByteArr = BytesIO()
    open_image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()

    results = predictor.classify_image(project.id, publish_iteration_name, imgByteArr)
    return results




def predictFromImageUrl(testing_image_urls, file_name):
    '''
    Takes a list of image urls, and saves a csv and html file giving the probability of each photo being a cane toad
    :param testing_image_urls: a list of image urls, where each element in the list is a list of 5 elements; the
    url, the label/description of the image, the latitude and longitude, and the date
    :param file_name: directory for csv and html file to be placed in.
    :return:
    '''

    # get prediction percentages
    image_predictions = []
    for url,species, lat, long, date in testing_image_urls:
        # initialise list of probabilities for each test and the corresponding image
        percentages = [-1]
        image_coords = [None]

        # first using google cloud to crop image into regions of interest:
        sections, tags = cropImage(url, return_coords=True)
        if len(sections)>0:
            for crop,coords in sections:
                # if cane toad identified, break out of loop
                #if max(percentages)>0.95:
                #    break
                results = predictFromImageFile(crop)
                for tag in results.predictions:
                    if tag.tag_name == 'caneToad':
                        percentages.append(tag.probability)
                        image_coords.append(coords)



        # if cane toad still not identified try whole photo
        if True: #max(percentages) < 0.95:
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

                    results = predictFromImageFile(img)


                # if image actually doesn't exist
                except OSError:
                    results = None

                else:
                    for tag in results.predictions:
                        if tag.tag_name == 'caneToad':
                            percentages.append(tag.probability)
                            # coordinates corresponding to whole image
                            image_coords.append('NA')
            else:
                for tag in results.predictions:
                    if tag.tag_name == 'caneToad':
                        percentages.append(tag.probability)
                        # coordinates corresponding to whole image
                        image_coords.append('NA')

        # get image or crop corresponding to highest probability
        coords = image_coords[np.argmax(percentages)]
        uncroppedProb = percentages[image_coords.index('NA')]

        image_predictions.append([url, coords, species, uncroppedProb, max(percentages), lat, long, date])




    # get in ascending order of probability
    #sorted_predictions = sorted(image_predictions, key=lambda tup: tup[3])
    # not sorting for now
    sorted_predictions=image_predictions


    # write new file with image urls and prediction percentages
    with open('predictions/'+file_name+'.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter = ',')
        wr.writerows([['url','crop','source','prediction','lat','long','date']])
        wr.writerows(sorted_predictions)

    # write html file
    with open('predictions/'+file_name+'.html', 'w') as myfile:
        myfile.write('<!doctype html> <html> <head> <meta charset="UTF-8"> <title>Untitled Document</title> </head>  <body><table>')
        myfile.write('<tr><th>Image</th><th>Best crop</th><th>Source</th><th>Uncropped cane toad prob</th><th>Best cropped cane toad prob</th></tr>')
        for url, coords, species, uncroppedProb, percentage, lat, long, date in sorted_predictions:
            myfile.write('<tr>')
            myfile.write("<td><img src='"+ url + "' width='250' alt=''/></td>")

            # crop image if needed
            if coords!='NA':
                x1, y1, x2, y2, ratio = coords
                height = (y2-y1)/(x2-x1)*250/ratio
                outsideWidth = 250/(x2-x1)
                marginLeft = x1*outsideWidth
                outsideHeight = height/(y2-y1)
                marginTop = y1*outsideHeight
                string = '<div style="width:250px; height:'+str(height)+'px; overflow: hidden; position: relative;"><img src="'\
                         +url+'" style = "width:'+str(outsideWidth)+'px; height:'+str(outsideHeight)+'px; margin-left:-'+str(marginLeft)\
                         +'px; margin-top:-'+str(marginTop)+'px; position: absolute">'
                string += '</div>'

                myfile.write("<td>"+string+"</td>")
            else:
                myfile.write("<td><img src='" + url + "' width='250' alt=''/></td>")

            myfile.write("<td>"+species+"</td>")
            myfile.write("<td>"+str(uncroppedProb)+"</td>")
            myfile.write("<td>"+str(percentage)+"</td>")
            myfile.write('</tr>')





# to use for photos that werent in training
def predict_from_csv():
    # initialise list to store image urls from file
    testing_image_urls = []
    with open('predictions/binaryAll.csv', 'r') as myfile:
        for url in myfile:
            url, species = url.split(',')
            testing_image_urls.append([url, species, 'NA', 'NA', 'NA'])