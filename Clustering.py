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

import json
import urllib.parse
import requests




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
            if len(image_labels)>-1:
                if label.description not in image_labels:
                    '''
                    # using google knowledge graph to try to gain insight into labels

                    api_key = open('api-key.txt').read()
                    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
                    params = {
                        'ids': label.mid,
                        'key': api_key,
                    }
                    url = f'{service_url}?{urllib.parse.urlencode(params)}'
                    response = requests.get(url, verify=False)
                    json_response = json.loads(response.text)

                    if json_response['itemListElement'] and len(json_response['itemListElement'][0]['result']['@type'])>1:
                        print(json_response['itemListElement'][0]['result']['@type'])'''
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
    distances = {}
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
        distances[1 + i + len(to_merge)] = merge[2]
        # ^ you could optionally store other info here (e.g distances)
    return clusters, distances




class comparing_clusters:
    def __init__(self, filterSource = 'ala'):
        # get ala images
        url_and_tags = getTagsFromPredictions(filterSource, return_note=True)

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
        for url, tags, coords, prediction, reid, note in url_and_tags:
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
                if label.description not in image_labels:
                    image_labels.append(label.description)


            self.AlaImageLabels[url] = image_labels

    def comparisonDict(self, url_and_tags_comparison):
        self.urlDict = {}
        # dictionay of all image urls so that tags can be accessed easily
        for url, tags, coords, prediction, reid, note in url_and_tags_comparison:
            self.urlDict[url] = [tags, coords, prediction, reid, note]

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
                    if label.description not in image_labels:
                        image_labels.append(label.description)

                myDict[url] = image_labels

            self.CompareImageLabels[website_source] = myDict

    def getCompareImageDist(self, website_source):

        if website_source not in self.compareImageDist.keys():
            labelDict = self.CompareImageLabels[website_source]
            distanceDict = {}

            for url in labelDict.keys():
                distances = []
                label1 = labelDict[url]
                if len(label1):
                    for ala_url in self.AlaImageLabels:
                        # find proportion of keywords in each image that do not overlap
                        label2 = self.AlaImageLabels[ala_url]
                        if len(label2):
                            uniqueLabels = list(set(label1 + label2))
                            notCommonLabels = 2 * len(uniqueLabels) - len(label1) - len(label2)
                            distance = notCommonLabels / (len(label1) + len(label2))
                            distances.append(distance)
                            distanceDict[url] = np.median(distances)
                else:
                    distanceDict[url] = 1

            self.compareImageDist[website_source] = distanceDict

    def rms(self, urlList):
        url_and_tags = []
        # build url_and_tags from dictionary
        for url in urlList:
            tags, coords, prediction, reid, note = self.urlDict[url]
            url_and_tags.append([url, tags, coords, prediction, reid, note])

        tagsList = []
        for url, tags, coords, prediction, reid, note in url_and_tags:
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


    def minALAdistance(self, urlList, website_source):
        distances = []
        for url in urlList:
            try:
                distances.append(self.compareImageDist[website_source][url])
            except KeyError:
                pass
        return np.min(distances)

    def createAlaSpread(self):
        alaLabels = []
        for url in self.AlaImageLabels.keys():
            alaLabels+=self.AlaImageLabels[url]

        ALAlabels, ALAcounts = np.unique(alaLabels, return_counts=True)
        self.ALAlabelFreq = {}
        for index,label in enumerate(ALAlabels):
            self.ALAlabelFreq[label] = ALAcounts[index]/len(self.AlaImageLabels.keys())

    def compareToSpread(self, url, website_source):
        labels = self.CompareImageLabels[website_source][url]
        distanceFromAve = []
        for label in labels:
            try:
                freq = self.ALAlabelFreq[label]
            except KeyError:
                freq = 0
            distanceFromAve.append(1 - freq)
        return np.average(distanceFromAve)



    def containingAlaKeywords(self, urlList, website_source):
        alaLabels = []
        for url in self.AlaImageLabels.keys():
            alaLabels+=self.AlaImageLabels[url]

        ALAlabels, ALAcounts = np.unique(alaLabels, return_counts=True)
        sorted_indices = np.argsort(-ALAcounts)
        sorted_alaLabels = ALAlabels[sorted_indices]
        frequentLabels = sorted_alaLabels[0:10]


        # find proportion of images that have a label within ala's frequent labels
        count = 0
        for url in urlList:
            labels = self.CompareImageLabels[website_source][url]

            for label in labels:
                if label in frequentLabels:
                    print(label)
                    count+=1
                    break

        return count/len(urlList)


    def proportionCaneToads(self, urlList):
        verified_urls = []
        for url in urlList:
            tags, coords, prediction, reid, note = self.urlDict[url]
            if reid=='T' or reid =='PT':
                verified_urls.append(url)
        try:
            return(len(verified_urls)/len(urlList))
        except ZeroDivisionError:
            return(0)

    def plot(self, dataframe, source, filtered_urls, tree, distance_tree):

        distMat = dataframe.values

        condDistMat = scipy.spatial.distance.squareform(distMat)
        clustered = scipy.cluster.hierarchy.ward(condDistMat)

        plt.clf()

        dn = scipy.cluster.hierarchy.dendrogram(clustered, color_threshold=0, labels=list(dataframe.columns),
                                                above_threshold_color='lightblue', distance_sort = 'ascending',
                                                get_leaves=True)







        for i, d in zip(dn['icoord'], dn['dcoord']):

            for dist_key in distance_tree.keys():
                if d[1] == distance_tree[dist_key]:
                    urls = tree[dist_key]
                    if len(urls)>40:
                        labels = []
                        for url in urls:
                            labels += alaComparison.CompareImageLabels[source][url]
                        unique, counts = np.unique(labels, return_counts=True)
                        most_common_label = unique[np.argmax(counts)]
                        print(most_common_label)

                        x = 0.5 * sum(i[1:3])
                        y = d[1]
                        #plt.plot(x, y, 'bo')
                        plt.annotate("%s %d%%" % (most_common_label, int(100*max(counts)/len(urls))), (x, y), xytext=(0, -8),
                                     textcoords='offset points',
                                     va='top', ha='center')



        ax = plt.gca()
        xlbls = ax.get_xticklabels()
        for lbl in xlbls:
            lbl.set_fontsize('x-large')

            # set colour based on verified to be cane toad
            for url in self.urlDict.keys():
                tags, coords, prediction, reid, note = self.urlDict[url]
                if lbl.get_text() == url:
                    #st=r'{'+ note+'}'


                    if reid == 'T' or reid == 'PT':
                        st = r'\textcolor{green}{-}'
                    else:
                        st = r'\textcolor{red}{-}'

                    if note!= '':
                        st += r'\textcolor{yellow}{-}'
                    elif note!= 'xTOAD RACE':
                        st += r'\textcolor{purple}{-}'
                    if note == 'xTOAD RACE':
                        st += r'\textcolor{gray}{-}'


                    

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

with open('./clustering/alaComparison.obj', 'rb') as pickle_file:
    alaComparison = pickle.load(pickle_file)


for source in ['flickr','twitter','instagram_all', 'reddit', 'inaturalist']:

    print(source)


    '''url_and_tags = getTagsFromPredictions(source)
    # remove any images without tags
    url_and_tags_new = []
    for index, element in enumerate(url_and_tags):
        if len(element[1] ) >-1:
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

    dataframe = distanceMatrix(urlList)
    dataframe.to_pickle("./clustering/"+source+"_distance.pkl")
    continue'''

    dataframe = pd.read_pickle(("./clustering/"+source+"_distance.pkl"))




    tree, distance_tree = create_tree(dataframe)

    # input url_and_tags from the source to build dictionary (don't need to preprocess as only urls in clusters are used)
    alaComparison.comparisonDict(getTagsFromPredictions(source, return_note=True))
    alaComparison.getCompareImageLabels(source)
    alaComparison.getCompareImageDist(source)


    # save object
    file = open("./clustering/alaComparison.obj", 'wb')
    pickle.dump(alaComparison, file)




    filtered_urls = []
    # remove last cluster from the list (as this is just everything
    keys = list(tree.keys())
    keys = keys[0:-1]
    for key in keys[0:dataframe.shape[0]]:
        sub_cluster = tree[key]
        rms = alaComparison.rms(sub_cluster)
        '''print('image list')
        print(len(sub_cluster))
        print('prop cane toads')
        print(alaComparison.proportionCaneToads(sub_cluster))
        print('average counts')'''

        if alaComparison.averageALAdistance(sub_cluster, source) < 0.95:
            filtered_urls += sub_cluster

    filtered_urls = []
    for sub_cluster in dataframe.columns:
        sub_cluster = [sub_cluster]
        if alaComparison.averageALAdistance(sub_cluster, source) < 0.80:
            filtered_urls += sub_cluster

    alaComparison.createAlaSpread()
    filtered_urls = []
    for sub_cluster in dataframe.columns:
        if alaComparison.compareToSpread(sub_cluster, source) < 0.80:
            filtered_urls.append(sub_cluster)


        #print(alaComparison.averageALAdistance(sub_cluster, source))
    filtered_urls = list(set(filtered_urls))
    #print(alaComparison.rms(filtered_urls))
    print(dataframe.shape[0])
    print(len(filtered_urls))
    print(alaComparison.proportionCaneToads(dataframe.columns))
    print(alaComparison.proportionCaneToads(filtered_urls))




    alaComparison.plot(dataframe, source, filtered_urls, tree, distance_tree)


    # save object
    file = open("./clustering/alaComparison.obj", 'wb')
    pickle.dump(alaComparison, file)




