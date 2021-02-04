import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/hannahburke/Documents/CaneToadMachineLearning/keys.json"
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from io import BytesIO
# Instantiates a client
client = vision.ImageAnnotatorClient()

import json
import urllib.parse
import requests
from createTagsFiles import getTagsFromPredictions
import csv



def getLabelsFromPredictions(file_name, return_note=False):
    url_and_labels = []
    with open('predictions/reid/' + file_name +'_labels.csv', "r") as csv_file:
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
    for website in ['ala', 'flickr', 'inaturalist', 'instagram_all', 'reddit', 'twitter']:

        url_and_tags = getTagsFromPredictions(website, return_note=True)
        url_and_labels = []

        for url, tags, coords, prediction, reid, note in url_and_tags:

            source = types.ImageSource(image_uri=url)

            image = types.Image(source=source)

            # Performs label detection on the image file
            response = client.label_detection(image=image)

            labels = response.label_annotations

            image_labels = []
            for label in labels:
                if len(image_labels) > -1:
                    if label.description not in image_labels:
                        image_labels.append(label.description)

            # use the image provided it had at least one label
            if len(image_labels) > 0:
                url_and_labels.append([url, tags, coords, prediction, reid, note, image_labels])


        # write new file with image urls and prediction percentages
        with open('predictions/reid/' + website +'_labels.csv', 'w') as myfile:
            wr = csv.writer(myfile, delimiter=',')
            wr.writerows(url_and_labels)


