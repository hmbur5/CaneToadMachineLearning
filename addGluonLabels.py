import mxnet as mx
import gluoncv

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
import cv2
import numpy as np


def getGluonFromPredictions(file_name, return_note=False):
    url_and_labels = []
    with open('predictions/german_wasp/' + file_name +'_gluon.csv', "r") as csv_file:
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
    for website in ['random']:
        print(website)

        url_and_tags = getTagsFromPredictions(website, return_note=True)
        url_and_labels = []

        for url, tags, coords, prediction, reid, note in url_and_tags:

            # you may modify it to switch to another model. The name is case-insensitive
            model_name = 'ResNet50_v1d'
            # download and load the pre-trained model
            net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
            # load image
            try:
                url_file = url.replace('/', '').replace(':','')
                fname = mx.test_utils.download(url, fname='instagram_images/'+url_file, overwrite=True)
            except Exception as e:
                print(e)
                print(url)
                continue
            try:
                img = mx.ndarray.array(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(e)
                print(url)
            # apply default data preprocessing
            transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
            # run forward pass to obtain the predicted score for each class
            pred = net(transformed_img)
            # map predicted values to probability by softmax
            prob = mx.nd.softmax(pred)[0].asnumpy()
            # find the 5 class indices with the highest score
            ind = mx.nd.argsort(pred, is_ascend=0)[0].astype('int').asnumpy().tolist()
            # print the class name and predicted probability
            image_labels = []
            for i in range(len(ind)):
                # add all labels with probability >10% then break
                if prob[ind[i]] <0.1:
                    break
                image_labels.append(net.classes[ind[i]])


            # use the image provided it had at least one label
            if len(image_labels) > 0:
                url_and_labels.append([url, tags, coords, prediction, reid, note, image_labels])


        # write new file with image urls and prediction percentages
        with open('predictions/german_wasp/' + website +'_gluon.csv', 'w') as myfile:
            wr = csv.writer(myfile, delimiter=',')
            wr.writerows(url_and_labels)




