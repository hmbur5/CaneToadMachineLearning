from addGoogleLabels import getLabelsFromPredictions
from createTagsFiles import getTagsFromPredictions
import urllib.request
from PIL import Image
import urllib.parse
import numpy as np
import pandas as pd
import scipy
import scipy.spatial
import scipy.cluster
import random
import matplotlib.pyplot as plt
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/hannahburke/Documents/CaneToadMachineLearning/keys.json"
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
# Instantiates a client
client = vision.ImageAnnotatorClient()

plotClustersBool = False

if plotClustersBool:
    import matplotlib
    from matplotlib import rc
    from matplotlib.backends.backend_pgf import FigureCanvasPgf

    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{color}')
    from createTagsFiles import getTagsFromPredictions

    pgf_with_latex = {
        "pgf.preamble": r'\usepackage{color}'
    }
    matplotlib.rcParams.update(pgf_with_latex)


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


def plotCluster(dataframe, tree, distance_tree, inCluster, title):

    global url_and_tags

    distMat = dataframe.values

    condDistMat = scipy.spatial.distance.squareform(distMat)
    clustered = scipy.cluster.hierarchy.ward(condDistMat)

    plt.clf()
    dn = scipy.cluster.hierarchy.dendrogram(clustered, color_threshold=0, labels=list(dataframe.columns),
                                            above_threshold_color='lightblue', distance_sort='ascending',
                                            get_leaves=True)

    for i, d in zip(dn['icoord'], dn['dcoord']):

        ''' # if wanting to add most common label among clusters (this uses too much memory for large image sets)
        for dist_key in distance_tree.keys():
            if d[1] == distance_tree[dist_key]:
                urls = tree[dist_key]
                if len(urls) > 40:
                    labels = []
                    for url in urls:
                        for item in url_and_tags:
                            if item[0] == url:
                                labels += item[1]
                    unique, counts = np.unique(labels, return_counts=True)
                    most_common_label = unique[np.argmax(counts)]


                    x = 0.5 * sum(i[1:3])
                    y = d[1]
                    # plt.plot(x, y, 'bo')
                    plt.annotate("%s %d%%" % (most_common_label,100*max(counts)/len(urls)), (x, y), xytext=(0, -2),
                                 textcoords='offset points',
                                 va='top', ha='center')
        '''

    ax = plt.gca()

    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        lbl.set_fontsize('x-large')

        if lbl.get_text() in inCluster:
            st=r'\textcolor{blue}{-}'
        else:
            st = r'\textcolor{yellow}{-}'
        # set colour based on verified to be cane toad

        for item in url_and_tags:
            if lbl.get_text() == item[0]:
                try:
                    reid = item[4]
                except IndexError:
                    # example url
                    pass
                    st += r'\textcolor{black}{-}'
                else:
                    if reid == 'T' or reid == 'PT':
                        st += r'\textcolor{green}{-}'
                    else:
                        st += r'\textcolor{red}{-}'

        lbl.set_text(st)

    ax.set_xticklabels(xlbls)
    plt.title(title)
    plt.autoscale()
    #plt.show()
    plt.savefig('./clustering/iteratively/' + title + '.pdf', bbox_inches='tight')


def identifyCluster(dataframe, clustering_urls, example_url, iteration, website):
    '''
    Takes the entire image set dataframe, reduces it to the remaining urls that require clustering, and returns two lists
    of image urls, where the first one is the largest cluster containing the example_url, where the distance within cluster is
    <1/3 the maximum distance, and the second is the remaining images.
    Also plots the dendrogram of the clustering, with colour coding.
    :param dataframe: matrix of distances between each pair of images, with image urls as column headers.
    :param clustering_urls: subset of urls that require clustering
    :param example_url: url of example image to comare to
    :param iteration: number of iteration in iterative clustering (for title of cluster plot)
    :param website: website source of images (for title of cluster plot)
    :return: a list of two lists, the first being largest cluster that the example url falls into, where the
    distance with cluster <1/3 the maximum distance, and the second being the remaining images.
    '''
    indices = [dataframe.columns.get_loc(c) for c in clustering_urls if c in dataframe]
    subsetDF = dataframe.loc[indices, clustering_urls]
    tree, distance_tree = create_tree(subsetDF)
    keys = list(tree.keys())

    maxDist = distance_tree[keys[-1]]
    inCluster = []
    # finding cluster with example image that is below distance threshold
    for distThresh in [1, 1/3*maxDist]:
        for key in keys[0:-2]:
            sub_cluster = tree[key]
            if len(sub_cluster)>len(inCluster):
                if example_url in sub_cluster and distance_tree[key]<distThresh:
                    inCluster=sub_cluster
        if len(inCluster)>1:
            break

    # get remaining images
    outCluster = tree[keys[-1]]
    for image in inCluster:
        outCluster.remove(image)

    if plotClustersBool:
        plotCluster(subsetDF, tree, distance_tree, inCluster, website+' iteration '+str(iteration))

    return inCluster, outCluster




if __name__ == '__main__':

    example_labels = []
    while len(example_labels) == 0:
        # add example image
        ''''''
        example_url = input('Enter url of example image: ')

        source = types.ImageSource(image_uri=example_url)

        image = types.Image(source=source)

        # Performs label detection on the image file
        response = client.label_detection(image=image)

        labels = response.label_annotations

        for label in labels:
            if label.description not in example_labels:
                example_labels.append(label.description)

        '''
        response = client.object_localization(image=image)
        for tag in response.localized_object_annotations:
            example_labels.append(tag.name)
        example_labels = list(set(example_labels))
        '''

        print(example_labels)
        # use the image provided it had at least one label
        if len(example_labels) == 0:
            print('No labels were identified on google cloud vision. Try a different image. ')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    yVal = 0.2
    xMax = 0

    for website in ['flickr', 'inaturalist', 'twitter', 'reddit']:

        global url_and_tags
        url_and_tags = getLabelsFromPredictions(website)
        random.shuffle(url_and_tags)

        # remove any images without labels
        url_and_tags_new = []
        for index, element in enumerate(url_and_tags):
            if len(element[1]) > 0:
                url_and_tags_new.append([element[0], list(set(element[1])), element[2], element[3], element[4]])
        url_and_tags = url_and_tags_new

        # remove duplicate images
        url_and_tags_new = []
        open_images = []
        maxLabels = 0
        for index, element in enumerate(url_and_tags):
            image_url = element[0]
            try:
                img = urllib.request.urlopen(image_url)
                img = Image.open(img)
                if img not in open_images:
                    open_images.append(img)
                    url_and_tags_new.append(element)
                    if len(element[1])>maxLabels:
                        maxLabels = len(element[1])
                else:
                    pass
                    # print('duplicate')
            except urllib.error.HTTPError:
                print(image_url)
        url_and_tags = url_and_tags_new


        url_and_tags.append([example_url, example_labels])

        image_urls = []
        disMat = np.zeros((len(url_and_tags), len(url_and_tags)))
        for index1, item1 in enumerate(url_and_tags):
            image_urls.append(item1[0])

            for index2, item2 in enumerate(url_and_tags):
                label1 = item1[1]
                label2 = item2[1]
                # find proportion of keywords in each image that do not overlap
                uniqueLabels = list(set(label1 + label2))
                notCommonLabels = 2 * len(uniqueLabels) - len(label1) - len(label2)
                distance = notCommonLabels / (len(label1) + len(label2))

                #commonLabels = [value for value in label2 if value in label1]
                #distance = 1-len(commonLabels)/maxLabels

                disMat[index1, index2] = distance
                disMat[index2, index1] = distance

                if index1 == index2 and distance != 0:
                    print(label1)
                    print(label2)


        dataframe = pd.DataFrame(disMat, columns=image_urls)


        image_order = []


        labelsCommonDict = {}

        # add example image to each item in dictionary
        for index in range(len(example_labels)+1):
            labelsCommonDict[index] = [example_url]
        ##### first order images based on number of tags in common
        for item in url_and_tags:
            url = item[0]
            labels = item[1]
            commonLabels = [value for value in labels if value in example_labels]
            overlap = len(commonLabels)
            #to group everything
            #overlap = 0
            labelsCommonDict[overlap] = labelsCommonDict[overlap]+[url]


        for index in range(len(example_labels), -1, -1):
            clustering_urls = labelsCommonDict[index]

            # skip if there is nothing to compare to
            if len(clustering_urls)<=1:
                continue
            iteration = 0

            # iteratively clustering based on inCluster/outCluster compared to example url
            while True:
                iteration +=1
                inCluster, outCluster = identifyCluster(dataframe, clustering_urls, example_url, iteration, website)
                clustering_urls = outCluster+[example_url]

                if len(inCluster)==0:
                    image_order += outCluster
                    break

                inCluster.remove(example_url)

                # Add this cluster to removed images. Order in increasing distance from example image
                new_order_vals = []
                for url in inCluster:
                    new_order_vals.append(dataframe.loc[(np.shape(dataframe.values)[0] - 1), url])
                for index in np.argsort(new_order_vals):
                    image_order.append(inCluster[index])



        xVal = 0.05

        print(len(image_order))

        image_order.reverse()

        verification = []

        for url in image_order:
            reid = 'Unfound'
            for item in url_and_tags[0:-1]:
                if item[0]==url:
                    reid = item[4]
            if reid == 'Unfound':
                print(url)
                continue
            elif reid == 'T' or reid == 'PT':
                verification.append(1)
                ax1.text(xVal, yVal,
                         '|',
                         color="green")
            else:
                verification.append(0)
                ax1.text(xVal, yVal,
                         '|',
                         color="red")
            xVal += 0.07
        if xVal>xMax:
            xMax = xVal
        ax1.text(xVal / 2, yVal + 0.2, website, color='black')
        ax1.set_ylim(0, yVal+1)
        ax1.set_xlim(0, xMax+0.5)
        #yVal += 1.3
        print('done')


        # plotting percentage verified as move up list
        percentageVerified = []
        for index in range(len(verification)-1):
            portion = verification[index:-1]
            percentageVerified.append(sum(portion)/len(portion))
        ax2.plot(np.linspace(0, 1, num = len(percentageVerified)), percentageVerified)


        # using just distance from example image
        new_order_vals = []
        new_image_order = []

        for url in image_order:
            new_order_vals.append(dataframe.loc[(np.shape(dataframe.values)[0] - 1), url])
        for index in np.argsort(new_order_vals):
            new_image_order.append(image_order[index])
        new_image_order.reverse()


        xVal = 0.05

        verification = []

        for url in new_image_order:
            reid = 'Unfound'
            for item in url_and_tags[0:-1]:
                if item[0] == url:
                    reid = item[4]
            if reid == 'Unfound':
                print(url)
                continue
            elif reid == 'T' or reid == 'PT':
                verification.append(1)
                ax3.text(xVal, yVal,
                         '|',
                         color="green")
            else:
                verification.append(0)
                ax3.text(xVal, yVal,
                         '|',
                         color="red")
            xVal += 0.07
        if xVal > xMax:
            xMax = xVal
        ax3.text(xVal / 2, yVal + 0.2, website, color='black')
        ax3.set_ylim(0, yVal+1)
        ax3.set_xlim(0, xMax + 0.5)
        yVal += 0.8
        print('done')

        # plotting percentage verified as move up list
        percentageVerified = []
        for index in range(len(verification) - 1):
            portion = verification[index:-1]
            percentageVerified.append(sum(portion) / len(portion))
        ax4.plot(np.linspace(0, 1, num=len(percentageVerified)), percentageVerified, '--')



    ax2.legend(['flickr', 'inaturalist', 'twitter', 'reddit'])

    plt.show()





