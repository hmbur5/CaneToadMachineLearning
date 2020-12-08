import pandas as pd
import numpy as np
import scipy
import scipy.spatial
import scipy.cluster
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')

pgf_with_latex = {
    "pgf.preamble": r'\usepackage{xcolor}'
}
matplotlib.rcParams.update(pgf_with_latex)

import csv
import os
import ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/hannahburke/Documents/CaneToadMachineLearning/keys.json"
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from io import BytesIO
# Instantiates a client
client = vision.ImageAnnotatorClient()
from Filtering import filter


def distanceMatrix(image_urls):

    labelsList = []

    image_urls_labelled = []

    for url in image_urls:

        source = types.ImageSource(image_uri=url)

        image = types.Image(source=source)

        # Performs label detection on the image file
        response = client.label_detection(image=image)

        labels = response.label_annotations

        image_labels = []
        for label in labels:
            # get first 10 labels
            if len(image_labels)<10:
                if label.description not in image_labels:
                    image_labels.append(label.description)

        # use the image provided it had at least one label
        if len(image_labels)>0:
            image_urls_labelled.append(url)
            labelsList.append(image_labels)

    disMat = np.zeros((len(image_urls_labelled), len(image_urls_labelled)))
    for index1, label1 in enumerate(labelsList):
        for index2, label2 in enumerate(labelsList):

            # find proportion of keywords in each image that do not overlap
            uniqueLabels = list(set(label1+label2))
            notCommonLabels = 2*len(uniqueLabels) - len(label1) - len(label2)
            distance = notCommonLabels/(len(label1)+len(label2))

            disMat[index1, index2] = distance
            disMat[index2, index1] = distance

            if index1==index2 and distance!=0:
                print(label1)
                print(label2)




    dataframe = pd.DataFrame(disMat, columns=image_urls_labelled)
    print(dataframe)

    return(dataframe)




def cluster(dataframe, source):

    distMat = dataframe.values

    condDistMat = scipy.spatial.distance.squareform(distMat)
    clustered = scipy.cluster.hierarchy.ward(condDistMat)


    url_and_tags = getTagsFromPredictions(source)
    url_and_tags_filtered = filter(url_and_tags)
    filtered_urls = []
    for element in url_and_tags_filtered:
        filtered_urls.append(element[0])


    plt.clf()

    dn = scipy.cluster.hierarchy.dendrogram(clustered, color_threshold=0, labels=list(dataframe.columns))
    ax = plt.gca()
    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        lbl.set_fontsize('x-large')

        # set colour based on verified to be cane toad
        for element in url_and_tags:
            if lbl.get_text() == element[0]:
                reid = element[4]
                if reid == 'T' or reid == 'PT':
                    st = r'\textcolor{green}{-}'
                else:
                    st = r'\textcolor{red}{-}'

        # set letter based on whether it is filtered/unfiltered
        if lbl.get_text() in filtered_urls:
            st += r'\textcolor{blue}{-}'
        else:
            st += r'\textcolor{orange}{-}'

        lbl.set_text(st)


    ax.set_xticklabels(xlbls)
    plt.title(source)
    plt.autoscale()

    #plt.show()
    plt.savefig('./clustering/'+source+'.pdf', bbox_inches='tight')



from createTagsFiles import getTagsFromPredictions
for source in ['twitter','ala','instagram_new', 'twitter', 'flickr', 'reddit', 'inaturalist']:

    print(source)


    url_and_tags = getTagsFromPredictions(source)
    # remove any images without tags
    url_and_tags_new = []
    for index, element in enumerate(url_and_tags):
        if len(element[1] ) >0:
            url_and_tags_new.append(url_and_tags[index])
    url_and_tags = url_and_tags_new

    # remove duplicate images
    url_and_tags_new = []
    open_images = []
    for index, element in enumerate(url_and_tags):
        image_url = element[0]
        try:
            img = urllib.request.urlopen(image_url)
            img = Image.open(img)
            if img not in open_images:
                open_images.append(img)
                url_and_tags_new.append(element)
            else:
                pass
                #print('duplicate')
        except urllib.error.HTTPError:
            print(image_url)
    url_and_tags = url_and_tags_new


    urlList = []
    for element in url_and_tags:
        urlList.append(element[0])

    #dataframe = distanceMatrix(urlList)
    #dataframe.to_pickle("./clustering/"+source+"_distance.pkl")

    dataframe = pd.read_pickle(("./clustering/"+source+"_distance.pkl"))
    cluster(dataframe, source)


