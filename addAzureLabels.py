import os

from io import BytesIO

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time

import json
import urllib.parse
import requests
from createTagsFiles import getTagsFromPredictions
import csv
import math

subscription_key = "eb923ba89759461db59d8a5542f54569"
endpoint = "https://canetoadimageclassification1.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))




def getAzureTagsFromPredictions(file_name, file_dir, return_note=False):
    url_and_labels = []
    with open('predictions/'+file_dir+'/' + file_name +'_azure.csv', "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
            # skip over first line
            if lines[0]=='url':
                continue

            url = lines[0]
            tags = lines[1]
            if tags[0]!= '[':
                continue
            tags = tags[1:-1] # remove [ and ] from string
            if len(tags)==0:
                newTags = []
            else:
                tags = list(tags.split(", ")) # convert back to list
                # remove quotation marks from each string
                newTags = []
                for tag in tags:
                    newTags.append(tag[1:-1])
            # getting best crop
            coords = lines[2]
            if coords!='NA':
                coords = coords[1:-1]  # remove [ and ] from string
                coords = list(coords.split(", "))  # convert back to list
                for index, coord in enumerate(coords):
                    coords[index] = float(coord)
            prediction = float(lines[3])
            reid = lines[4]
            note = lines[5]
            labels = lines[6]
            if labels[0]!= '[':
                continue
            labels = labels[1:-1] # remove [ and ] from string
            if len(labels)==0:
                newLabels = []
            else:
                labels = list(labels.split(", ")) # convert back to list
                # remove quotation marks from each string
                newLabels = []
                for label in labels:
                    newLabels.append(label[1:-1])

            if return_note:
                url_and_labels.append([url, newLabels, coords, prediction, reid, note])
            else:
                url_and_labels.append([url, newLabels, coords, prediction, reid])

    return url_and_labels


if __name__ == '__main__':
    for website in ['instagram']:

        url_and_tags = getTagsFromPredictions(website, return_note=True)
        url_and_labels = []

        for url, tags, coords, prediction, reid, note in url_and_tags:

            # Call API with remote image
            try:
                labels = computervision_client.tag_image(url).tags
            except:
                try:
                    # resize and try again
                    img = urllib.request.urlopen(url)
                    img = Image.open(img)
                    height, width = img.size
                    currentSize = len(img.fp.read())
                    # these numbers could probably be better
                    maxDim = math.floor(max([width, height]) * 4194304 / currentSize / 3)
                    img.thumbnail((maxDim, maxDim))

                    img.save('resizing' + ".jpg", "JPEG")
                    img = open('resizing.jpg','rb')

                    labels = computervision_client.tag_image_in_stream(img).tags
                except urllib.error.HTTPError:
                    print(url)
                    continue

            image_labels = []
            for label in labels:
                if len(image_labels) > -1:
                    if label.name not in image_labels:
                        image_labels.append(label.name)

            # use the image provided it had at least one label
            if len(image_labels) > 0:
                url_and_labels.append([url, tags, coords, prediction, reid, note, image_labels])


        # write new file with image urls and prediction percentages
        with open('predictions/german_wasp/' + website +'_azure.csv', 'w') as myfile:
            wr = csv.writer(myfile, delimiter=',')
            wr.writerows(url_and_labels)


