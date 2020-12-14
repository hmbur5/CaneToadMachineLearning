import pandas as pd
import numpy as np
import scipy
import scipy.spatial
import scipy.cluster
import urllib.request
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')
from createTagsFiles import getTagsFromPredictions

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






def create_tree(dataframe):
    clusters = {}
    labels = dataframe.columns
    condDistMat = scipy.spatial.distance.squareform(dataframe.values)
    to_merge = scipy.cluster.hierarchy.ward(condDistMat)
    for i, merge in enumerate(to_merge):
        if merge[0] <= len(to_merge):
            # if it is an original point read it from the centers array
            a = [labels[int(merge[0])]]
        else:
            # other wise read the cluster that has been created
            a = clusters[int(merge[0])]

        if merge[1] <= len(to_merge):
            b = [labels[int(merge[1])]]
        else:
            b = clusters[int(merge[1])]
        # the clusters are 1-indexed by scipy
        clusters[1 + i + len(to_merge)] = a+b
        # ^ you could optionally store other info here (e.g distances)
    return clusters




class comparing_clusters:
    def __init__(self, filterSource = 'ala'):
        # get ala images
        url_and_tags = getTagsFromPredictions(filterSource)

        # remove any images without tags
        url_and_tags_new = []
        for index, element in enumerate(url_and_tags):
            if len(element[1]) > 0:
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
                    # print('duplicate')
            except urllib.error.HTTPError:
                print(image_url)
        url_and_tags = url_and_tags_new

        tagsListALA = []
        urlList = []
        for url, tags, coords, prediction, reid in url_and_tags:
            urlList.append(url)
            tagsListALA += list(set(tags))

        no_imagesALA = len(url_and_tags)

        ALAlabels, ALAcounts = np.unique(tagsListALA, return_counts=True)

        self.alaURLs = urlList
        self.ALAlabels = ALAlabels
        self.ALAcounts = ALAcounts
        self.tagsListALA = tagsListALA
        self.no_imagesALA = no_imagesALA
        self.CompareImageLabels = {}
        self.compareImageDist = {}


    def getAlaImageLabels(self):
        self.AlaImageLabels = {}
        for url in self.alaURLs:
            print(url)

            source = types.ImageSource(image_uri=url)

            image = types.Image(source=source)

            # Performs label detection on the image file
            response = client.label_detection(image=image)

            labels = response.label_annotations

            image_labels = []
            for label in labels:
                # get first 10 labels
                if len(image_labels) < 10:
                    if label.description not in image_labels:
                        image_labels.append(label.description)


            self.AlaImageLabels[url] = image_labels

    def comparisonDict(self, url_and_tags_comparison):
        self.urlDict = {}
        # dictionay of all image urls so that tags can be accessed easily
        for url, tags, coords, prediction, reid in url_and_tags_comparison:
            self.urlDict[url] = [tags, coords, prediction, reid]

    def getCompareImageLabels(self, website_source):
        if website_source not in self.CompareImageLabels.keys():
            myDict = {}

            for url in self.urlDict.keys():
                print(url)

                source = types.ImageSource(image_uri=url)

                image = types.Image(source=source)

                # Performs label detection on the image file
                response = client.label_detection(image=image)

                labels = response.label_annotations

                image_labels = []
                for label in labels:
                    # get first 10 labels
                    if len(image_labels) < 10:
                        if label.description not in image_labels:
                            image_labels.append(label.description)


                myDict[url] = image_labels
            self.CompareImageLabels[website_source] = myDict

    def getCompareImageDist(self, website_source):
        # the distance between an image and the set of ala images is
        # the proportion of all ala image labels that are not in that image


        if website_source not in self.compareImageDist.keys():
            labelDict = self.CompareImageLabels[website_source]
            distanceDict = {}

            allALAlabels = []
            for url in self.alaURLs:
                allALAlabels += self.AlaImageLabels[url]

            for url in labelDict.keys():
                labels = self.CompareImageLabels[website_source][url]
                count = 0
                for label in labels:
                    count += allALAlabels.count(label)

                distanceDict[url] = 1- count/len(allALAlabels)
            self.compareImageDist[website_source] = distanceDict

    def rms(self, urlList):
        url_and_tags = []
        # build url_and_tags from dictionary
        for url in urlList:
            tags, coords, prediction, reid = self.urlDict[url]
            url_and_tags.append([url, tags, coords, prediction, reid])

        tagsList = []
        for url, tags, coords, prediction, reid in url_and_tags:
            tagsList += list(set(tags))

        labels, counts = np.unique(tagsList, return_counts=True)
        for label in self.ALAlabels:
            labels = np.append(labels, label)
        labels = list(set(labels))

        counts = [0.0] * len(labels)
        ALAcounts = [0.0] * len(labels)
        for index, label in enumerate(labels):
            for tag in tagsList:
                if tag == label:
                    counts[index] += 1
            for tag in self.tagsListALA:
                if tag == label:
                    ALAcounts[index] += 1

        for index, count in enumerate(counts):
            ALAcounts[index] = ALAcounts[index] / self.no_imagesALA
            count = count / len(url_and_tags)
            try:
                counts[index] = count - ALAcounts[index]
            except IndexError:
                counts[index] = count

        abscounts = []
        for count in counts:
            abscounts.append(-abs(count))

        sorted_indices = np.argsort(abscounts)
        sortedCounts = []
        sortedLabels = []
        for i in sorted_indices:
            sortedCounts.append(counts[i])
            sortedLabels.append(labels[i])

        rms_error = np.sqrt(np.mean(np.square(sortedCounts)))
        return rms_error

    def averageALAdistance(self, urlList, website_source):
        distances = []
        for url in urlList:
            try:
                distances.append(self.compareImageDist[website_source][url])
            except KeyError:
                pass
        return np.average(distances)


    def proportionCaneToads(self, urlList):
        verified_urls = []
        for url in urlList:
            tags, coords, prediction, reid = self.urlDict[url]
            if reid=='T' or reid =='PT':
                verified_urls.append(url)
        try:
            return(len(verified_urls)/len(urlList))
        except ZeroDivisionError:
            return(0)

    def plot(self, dataframe, source, filtered_urls):

        distMat = dataframe.values

        condDistMat = scipy.spatial.distance.squareform(distMat)
        clustered = scipy.cluster.hierarchy.ward(condDistMat)

        plt.clf()

        dn = scipy.cluster.hierarchy.dendrogram(clustered, color_threshold=0, labels=list(dataframe.columns))
        ax = plt.gca()
        xlbls = ax.get_xticklabels()
        for lbl in xlbls:
            lbl.set_fontsize('x-large')

            # set colour based on verified to be cane toad
            for url in self.urlDict.keys():
                tags, coords, prediction, reid = self.urlDict[url]
                if lbl.get_text() == url:
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

        # plt.show()
        plt.savefig('./clustering/' + source + '.pdf', bbox_inches='tight')


from createTagsFiles import getTagsFromPredictions
#alaComparison = comparing_clusters()
#alaComparison.getAlaImageLabels()
#file = open("./clustering/alaComparison.obj", 'wb')
#pickle.dump(alaComparison, file)

with open('./clustering/alaComparison_orig.obj', 'rb') as pickle_file:
    alaComparison = pickle.load(pickle_file)


for source in ['flickr','twitter','instagram_new', 'flickr', 'reddit', 'inaturalist']:

    print(source)


    '''url_and_tags = getTagsFromPredictions(source)
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
        urlList.append(element[0])'''

    #dataframe = distanceMatrix(urlList)
    #dataframe.to_pickle("./clustering/"+source+"_distance.pkl")

    dataframe = pd.read_pickle(("./clustering/"+source+"_distance.pkl"))
    #clustered = cluster(dataframe, source, filteredUrls)

    #print(clustered)
    tree = create_tree(dataframe)

    # input url_and_tags from the source to build dictionary (don't need to preprocess as only urls in clusters are used)
    alaComparison.comparisonDict(getTagsFromPredictions(source))
    alaComparison.getCompareImageLabels(source)

    # save object
    file = open("./clustering/alaComparison.obj", 'wb')
    pickle.dump(alaComparison, file)


    filtered_urls = []
    for key in tree.keys():
        sub_cluster = tree[key]
        rms = alaComparison.rms(sub_cluster)
        '''print('image list')
        print(len(sub_cluster))
        print('prop cane toads')
        print(alaComparison.proportionCaneToads(sub_cluster))
        print('average counts')'''
        alaComparison.getCompareImageDist(source)
        if alaComparison.averageALAdistance(sub_cluster, source) < 0.70:
            filtered_urls += sub_cluster
        #print(alaComparison.averageALAdistance(sub_cluster, source))
    filtered_urls = list(set(filtered_urls))
    #print(alaComparison.rms(filtered_urls))
    print(dataframe.shape[0])
    print(len(filtered_urls))
    print(alaComparison.proportionCaneToads(filtered_urls))

    alaComparison.plot(dataframe, source, filtered_urls)


    # save object
    file = open("./clustering/alaComparison.obj", 'wb')
    pickle.dump(alaComparison, file)





