import csv
import urllib.request
from PIL import Image
import os, ssl

from createTagsFiles import getTagsFromPredictions
from addGoogleLabels import getLabelsFromPredictions
from addAzureLabels import getAzureTagsFromPredictions
from addGluonLabels import getGluonFromPredictions
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import csv
import urllib
from PIL import Image
from google.cloud import vision
from google.cloud.vision import types

client = vision.ImageAnnotatorClient()
import io
import math

# write new file with image urls and prediction percentages
with open('tags/TFD_sums.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(['classifier', 'website', 'proportion verified cane toad',
                 'weighted sum error compared to ala',
                 'rms error compared to ala'])

    dict_class = {}
    for species in ['cane_toad', 'german_wasp','camel']:
        dict_class[species] = {}
        for classifier, classifier_results in [['tags', getTagsFromPredictions], ['labels', getLabelsFromPredictions],
                                               ['azure', getAzureTagsFromPredictions],
                                               ['imagenet', getGluonFromPredictions]]:
            dict_class[species][classifier] = {}

            for source in ['ala', 'flickr', 'twitter', 'reddit', 'inaturalist', 'instagram', 'ala1', 'ala2',
                           'random']:
                # skip over instagram if necessary as i dont have these images
                if species != 'german_wasp' and source == 'instagram':
                    continue
                row_to_add = []
                row_to_add.append(classifier)

                print(source)
                row_to_add.append(source)

                # if comparing objects, use getTags, if comparing labels, use getLabels
                # returns same shape list just there are more detailed names in the tags list.
                website = source
                if source == 'ala1' or source == 'ala2':
                    website = 'ala'
                url_and_tags = classifier_results(website, species)

                if source == 'ala1':
                    url_and_tags = url_and_tags[::2]
                elif source == 'ala2':
                    url_and_tags = url_and_tags[1::2]

                if source == 'ala':
                    url_and_tags = url_and_tags[::2]

                # remove any images without tags
                url_and_tags_new = []
                for index, element in enumerate(url_and_tags):
                    if len(element[1]) > 0:
                        url_and_tags_new.append(url_and_tags[index])
                url_and_tags = url_and_tags_new

                '''
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
                            print('duplicate')
                    except urllib.error.HTTPError:
                        print(image_url)
                url_and_tags = url_and_tags_new'''

                # check probability
                CtagsList = []
                NtagsList = []
                image_url_list = []
                url_and_tags_canetoad = []

                for url, tags, coords, prediction, reid in url_and_tags:
                    if reid == 'Y' or reid == 'M' or reid == 'C' or reid == 'T' or reid == 'PT':
                        CtagsList += list(set(tags))
                    else:
                        NtagsList += list(set(tags))
                    if prediction > 0.90:
                        url_and_tags_canetoad.append([url, tags, coords, prediction, reid])
                    image_url_list.append(url)
                # print('proportion >90% cane toad probability')
                # print(len(url_and_tags_canetoad)/len(url_and_tags))
                # row_to_add.append("%.4f"%(len(url_and_tags_canetoad)/len(url_and_tags)))

                url_and_tags_canetoad = []
                for url, tags, coords, prediction, reid in url_and_tags:
                    if reid == 'Y' or reid == 'M' or reid == 'C' or reid == 'T' or reid == 'PT':
                        url_and_tags_canetoad.append([url, tags, coords, prediction, reid])
                print('proportion verified cane toad')
                print(len(url_and_tags_canetoad) / len(url_and_tags))
                prop_verified = len(url_and_tags_canetoad) / len(url_and_tags)
                row_to_add.append("%.4f" % (len(url_and_tags_canetoad) / len(url_and_tags)))

                # create histogram
                tagsList = CtagsList + NtagsList
                labels, counts = np.unique(tagsList, return_counts=True)
                # sort in descending order
                sorted_indices = np.argsort(-counts)
                counts = counts[sorted_indices]
                labels = labels[sorted_indices]

                # split counts into cane toad and not cane toad
                Ccounts = [0.0] * len(counts)
                Ncounts = [0.0] * len(counts)
                for index, label in enumerate(labels):
                    for tag in CtagsList:
                        if tag == label:
                            Ccounts[index] += 1
                    for tag in NtagsList:
                        if tag == label:
                            Ncounts[index] += 1

                # counts = counts[0:25]
                # labels = labels[0:25]
                ticks = range(len(counts))
                # normalise to proportion of total number of images
                counts = counts / len(image_url_list)
                # normalise so peak is at 1
                Ccounts = Ccounts / max(counts)
                # counts = counts / max(counts)

                plt.clf()
                plt.bar(ticks[0:25], counts[0:25], align='center', color='orange')
                # plt.bar(ticks[0:25], Ccounts[0:25], align='center', color='green')
                plt.xticks(ticks[0:25], labels[0:25], rotation='vertical')
                plt.gcf().subplots_adjust(bottom=0.35)
                plt.title(source)
                plt.title


                # if species == 'cane_toad' and (source == 'ala2' or source=='ala1'):
                # plt.show()

                # plt.show()

                # creating exponential fit
                def func(t, b):
                    return np.exp(-b * t)


                # popt, pcov = curve_fit(func, ticks, counts)
                xx = np.linspace(0, len(ticks[0:25]), 1000)
                # yy = func(xx, *popt)/max(counts)
                # print('rate of decay:')
                # print(popt)
                # row_to_add.append("e^( -%.2f * n)"%(popt[0]))
                # plt.plot(xx,yy,'b')
                # plt.show()

                tagsList = []
                for url, tags, coords, prediction, reid in url_and_tags:
                    for tag in list(set(tags)):
                        tagsList.append([tag, 1 / len(list(set(tags)))])

                # compare distributions
                if source == 'instagram_all':
                    tagsListInstagram = tagsList
                    no_imagesInstagram = len(url_and_tags)
                if source == 'twitter':
                    tagsListTwitter = tagsList
                    no_imagesTwitter = len(url_and_tags)
                if source == 'reddit':
                    tagsListReddit = tagsList
                    no_imagesReddit = len(url_and_tags)
                if source == 'inaturalist':
                    tagsListInaturalist = tagsList
                    no_imagesInaturalist = len(url_and_tags)
                if source == 'flickr':
                    tagsListFlickr = tagsList
                    no_imagesFlickr = len(url_and_tags)
                if source == 'ala':
                    tagsListALA = tagsList
                    no_imagesALA = len(url_and_tags)
                    row_to_add += ['0', '0']
                elif source == 'ala1':
                    tagsListALA = tagsList
                    no_imagesALA = len(url_and_tags)
                    row_to_add += ['0', '0']

                else:
                    ALAlabels, ALAcounts = np.unique(np.array(tagsListALA)[:, 0], return_counts=1)
                    # sort in descending order

                    labels, counts = np.unique(np.array(tagsList)[:, 0], return_counts=1)
                    for label in ALAlabels:
                        labels = np.append(labels, label)
                    labels = list(set(labels))

                    counts = [0.0] * len(labels)
                    ALAcounts = [0.0] * len(labels)
                    for index, label in enumerate(labels):
                        for tag, significance in tagsList:
                            if tag == label:
                                counts[index] += 1  # significance
                        for tag, significance in tagsListALA:
                            if tag == label:
                                ALAcounts[index] += 1  # significance

                    for index, count in enumerate(counts):
                        ALAcounts[index] = ALAcounts[index] / no_imagesALA
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

                    ticks = range(len(sortedCounts))
                    plt.clf()
                    plt.bar(ticks[0:25], sortedCounts[0:25], align='center')
                    plt.xticks(ticks[0:25], sortedLabels[0:25], rotation='vertical')
                    plt.title(classifier + ' TFDs using ' + source)
                    plt.ylim([-0.5, 0.5])
                    # if species=='cane_toad' and source=='ala2':
                    #plt.show()

                    sum_error = 0
                    print(sortedCounts)
                    for val in sortedCounts:
                        if val > 0:
                            # pass
                            sum_error += val
                        else:
                            sum_error -= val
                    row_to_add.append('%.4f' % (sum_error))
                    print('sum error from ala')
                    print(sum_error)
                    rms_error = np.sqrt(np.mean(np.square(sortedCounts)))
                    row_to_add.append('%.4f' % (rms_error))
                    print('RMS error from ala')
                    print(rms_error)

                    dict_class[species][classifier][source] = [sum_error, prop_verified]

                # continue

                # minumum number of tags to cover 90% of images
                # print('number of tags covering 90% of images:')
                covered = []
                for index in range(len(labels)):
                    if len(covered) > 0.90 * len(url_and_tags):
                        break
                    tag = labels[index]
                    for url, tags, coords, prediction, reid in url_and_tags:
                        if tag in tags and url not in covered:
                            covered.append(url)
                # row_to_add.append(index)
                # print(index)

                # number of images frog and animal cover
                # print('number of images frog and animal cover:')
                covered = []
                for tag in ['Frog', 'Animal']:
                    for url, tags, coords, prediction, reid in url_and_tags:
                        if tag in tags and url not in covered:
                            covered.append(url)
                # row_to_add.append("%.4f"%(len(covered)/len(url_and_tags)))
                # print(len(covered)/len(url_and_tags))

                # percentage of images with multiple tags:
                # print('proportion of some tags')
                url_and_tags_multiple = []
                url_and_tags_some = []
                for url, tags, coords, prediction, reid in url_and_tags:
                    if len(tags) >= 1:
                        url_and_tags_some.append([url, tags])
                    if len(tags) > 1:
                        url_and_tags_multiple.append([url, tags])
                # print(len(url_and_tags_some)/len(url_and_tags))
                # row_to_add.append("%.4f"%(len(url_and_tags_some)/len(url_and_tags)))
                print('proportion of multiple tags')
                print(len(url_and_tags_multiple) / len(url_and_tags))
                # row_to_add.append("%.4f"%(len(url_and_tags_multiple)/len(url_and_tags)))

                animals = ['Animal', 'Frog', 'Crab', 'Dog', 'Moths and butterflies', 'Snake', 'Bird', 'Turtle',
                           'Scorpion',
                           'Worm', 'Lizard', 'Insect', 'Fish', 'Oyster', 'Tortoise', 'Sea turtle', 'Cat', 'Sheep',
                           'Snail']

                # percentage of images with multiple animal tags:
                url_and_tags_multiple_animals = []
                url_and_tags_animal_and_human = []
                url_and_tags_animals = []
                missing = []

                for url, tags, coords, prediction, reid in url_and_tags:
                    animalsTags = [x for x in tags if x in animals]
                    missing += [x for x in tags if x not in animals]
                    if len(animalsTags) >= 1:
                        url_and_tags_animals.append([url, animalsTags])
                        if 'Person' in tags:
                            url_and_tags_animal_and_human.append([url, animalsTags + ['Person']])
                    if len(animalsTags) > 1:
                        url_and_tags_multiple_animals.append([url, animalsTags])

                wr.writerow(row_to_add)

                # comparing false positives to false negatives
                thresholds = np.linspace(0, 1, 100)
                thresholds_valid = []
                false_pos_thresh = []
                false_neg_thresh = []
                percentage_frog_animal_thresh = []
                verified_thresh = []
                exponents_thresh = []
                for thresh in thresholds:
                    false_pos = []
                    false_neg = []
                    positives = []
                    negatives = []
                    for url, tags, coords, prediction, reid in url_and_tags:
                        if reid == 'T' or reid == 'PT':
                            positives.append(url)
                            if prediction < thresh:
                                false_neg.append(url)
                        else:
                            negatives.append(url)
                            if prediction > thresh:
                                false_pos.append(url)
                    # false_pos_thresh.append(len(false_pos)/len(negatives))
                    # false_neg_thresh.append(len(false_neg)/len(positives))

                    # creating a set of images that are above the given threshold
                    above_thresh_set = []
                    verified = []
                    for url, tags, coords, prediction, reid in url_and_tags:
                        if prediction > thresh:
                            above_thresh_set.append([url, tags, coords, prediction, reid])

                    # number of images frog and animal cover
                    covered = []
                    tagsList = []
                    for tag in ['Frog', 'Animal']:
                        for url, tags, coords, prediction, reid in above_thresh_set:
                            if tag in tags and url not in covered:
                                covered.append(url)

                    for url, tags, coords, prediction, reid in above_thresh_set:
                        # check which ones are verified
                        if reid == 'T' or reid == 'PT':
                            verified.append(url)
                        # tags for histogram
                        tagsList += list(set(tags))

                    if len(above_thresh_set) > 0:
                        thresholds_valid.append(thresh)
                        verified_thresh.append(len(verified) / len(above_thresh_set))
                        percentage_frog_animal_thresh.append(len(covered) / len(above_thresh_set))

                        # get exponential
                        labels, counts = np.unique(tagsList, return_counts=1)
                        sorted_indices = np.argsort(-counts)
                        counts = counts[sorted_indices]
                        counts = counts / max(counts)
                        ticks = range(len(counts))
                        popt, pcov = curve_fit(func, ticks, counts)
                        exponents_thresh.append(popt[0])

                # plt.plot(thresholds, false_pos_thresh)
                # plt.plot(thresholds, false_neg_thresh)
                plt.title(source)
                plt.legend(['false positives', 'false negatives'])
                plt.xlabel('Probability threshold')
                plt.ylabel('Proportion of wrongly classified images')
                # plt.show()

                plt.plot(thresholds_valid, percentage_frog_animal_thresh)
                plt.plot(thresholds_valid, exponents_thresh)
                plt.plot(thresholds_valid, verified_thresh)
                plt.title(source)
                plt.legend(['percentage frog and animal covered', 'rate of decay', 'percentage verified as cane toad'])
                plt.xlabel('Probability threshold')
                plt.ylabel('Proportion of images out of images above the threshold')
                # plt.show()

# exit(-1)

# errors of each website
labels = []
countsDict = {}

for source in ['ala', 'flickr', 'twitter', 'reddit', 'instagram', 'inaturalist']:
    if source == 'ala':
        tagsList = tagsListALA
        no_images = no_imagesALA
    if source == 'instagram_all':
        tagsList = tagsListInstagram
        no_images = no_imagesInstagram
    if source == 'flickr':
        tagsList = tagsListFlickr
        no_images = no_imagesFlickr
    if source == 'twitter':
        tagsList = tagsListTwitter
        no_images = no_imagesTwitter
    if source == 'reddit':
        tagsList = tagsListReddit
        no_images = no_imagesReddit
    if source == 'inaturalist':
        tagsList = tagsListInaturalist
        no_images = no_imagesInaturalist

    labels_new, counts = np.unique(np.array(tagsList)[:, 0], return_counts=1)
    for label in labels_new:
        labels = np.append(labels, labels_new)
    labels = list(set(labels))

    counts = [0.0] * len(labels)
    for index, label in enumerate(labels):
        for tag, significance in tagsList:
            if tag == label:
                counts[index] += significance

    for index, count in enumerate(counts):
        counts[index] = count / no_images

    for index, label in enumerate(labels):
        try:
            countsDict[label].append([counts[index], source])
        except KeyError:
            countsDict[label] = [[counts[index], source]]

alaComparison = {}
allCounts = []
allLabels = []
for label in countsDict.keys():
    list = countsDict[label]
    for counts, source in list:
        if source == 'ala':
            alaComparison[label] = counts
    # if there were no counts of this tag
    if label not in alaComparison:
        alaComparison[label] = 0

    # convert all other counts to errors
    for index, tuple in enumerate(list):
        list[index][0] = tuple[0] - alaComparison[label]

        allLabels.append(label)
        allCounts.append(list[index][0])

abscounts = []
for count in allCounts:
    abscounts.append(-abs(count))

sorted_indices = np.argsort(abscounts)
sortedCounts = []
sortedLabels = []

for i in sorted_indices:
    sortedCounts.append(allCounts[i])
    if allLabels[i] not in sortedLabels:
        sortedLabels.append(allLabels[i])

plt.clf()
plt.ylim([-0.75, 0.75])
for source in ['flickr', 'twitter', 'instagram', 'reddit', 'inaturalist']:
    counts = []
    for label in sortedLabels:
        for tuple in countsDict[label]:
            if tuple[1] == source:
                break
        if tuple[1] != source:
            counts.append(0)
        else:
            counts.append(tuple[0])

    ticks = range(len(counts))
    plt.scatter(ticks[0:25], counts[0:25])

plt.xticks(ticks[0:25], sortedLabels[0:25], rotation='vertical')

plt.gcf().subplots_adjust(bottom=0.35)
plt.ylabel('Tag frequency deviation from ALA')
plt.legend(['flickr', 'twitter', 'instagram', 'reddit', 'inaturalist'])
# \plt.show()


# make line and use to predict
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

with open('tags/covariance.csv', 'w') as myfile:
    plt.clf()
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(['observed', 'predicted', 'website', 'classifier', 'class', 'classifier_class', 'classifier_number'])
    for index2, species in enumerate(['cane_toad', 'german_wasp', 'camel']):
        # plt.clf()
        markers = ['o', 'v', 's', 'p', '*']
        colors = ['red', 'green', 'blue', 'orange']
        for index1, classifier in enumerate(['tags', 'labels', 'azure', 'imagenet']):
            x1 = dict_class[species][classifier]['ala2'][0]
            # x1 = 0
            x2 = dict_class[species][classifier]['random'][0]
            print(x1)
            print(x2)
            y1 = 1
            y2 = 0
            m = (y1 - y2) / (x1 - x2)
            c = (x1 * y2 - x2 * y1) / (x1 - x2)

            y_pred = []
            y_obs = []
            # plt.clf()
            for index, source in enumerate(['flickr', 'twitter', 'instagram', 'reddit', 'inaturalist']):
                if species != 'german_wasp' and source == 'instagram':
                    continue
                x = dict_class[species][classifier][source][0]
                print(x)
                y_pred.append([m * x + c])
                y_obs.append([dict_class[species][classifier][source][1]])
                # plt.plot(index2, dict_class[species][classifier][source][1]-(m*x+c), marker=markers[index], color=colors[index1])
                plt.scatter(x, [dict_class[species][classifier][source][1]], marker=markers[index],
                            color=colors[index1])
                # plt.scatter((x-x1)/(x2-x1), [dict_class[species][classifier][source][1]], marker=markers[index], color = colors[index1])
                # plt.scatter(m*x+c, dict_class[species][classifier][source][1])
                wr.writerow([dict_class[species][classifier][source][1], m * x + c, source, classifier, species,
                             'cfier' + str(index1 + 1) + '_species' + str(index2 + 1), index1])
            # error = np.subtract(y_pred,y_obs)
            # plt.plot(index2, np.sqrt(np.mean(np.square(error))), marker=markers[index1], color='black')

            plt.plot([x1, x2], [y1, y2], color=colors[index1])
            # plt.plot([0,1],[y1,y2], color='black')

            # plt.title('Observed proportion of desired images vs predicted')
            # plt.legend(['flickr','twitter','instagram','reddit', 'inaturalist'])
            # plt.xlabel('Predicted proportion of images similar to content of authoritative source')
            # plt.ylabel('Observed proportion of images similar to content of authoritative source')

            regression = linear_model.LinearRegression(fit_intercept=False).fit(y_pred, y_obs)
            print('Coefficient: %.2f'
                  % regression.coef_)
            # The coefficient of determination:
            print('R^2: %.2f'
                  % r2_score(y_pred, y_obs))

            # plt.annotate('Coefficient: %.2f\n R^2: %.2f'%  (regression.coef_, r2_score(y_pred, y_obs)), [0.8,0.3])
            # plt.show()
        #plt.title("Errors for predicted and observed values across different platforms, species and classifiers")
        plt.title("Line predicting number of images of " +species+ "s compared to weighted sum of TFD")
        # plt.plot([x1, x2], [y1, y2], color='black')
        # plt.scatter([x1, x2], [y1, y2], color='black')
        # plt.scatter([8], [m * 8 + c], color='red')
        # plt.plot([3],[0],color='white')
        # plt.xticks([0,1,2,3],['cane_toad', 'german_wasp','camel',''],rotation='vertical')
        #plt.xlabel('Species')
        #plt.ylabel('Difference between prediction and observed value of fraction of desired images')
        plt.xlabel("Weighted sum of TFD from ALA images")
        plt.ylabel("Fraction of images of "+species+"s in image set")
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

        handles = [f("s", colors[i]) for i in range(4)]
        handles += [f(markers[i], "k") for i in range(5)]

        labels = ['tags', 'labels', 'azure', 'imagenet'] + ['flickr', 'twitter', 'instagram', 'reddit', 'inaturalist']

        plt.legend(handles, labels, loc=1, framealpha=1)
        # plt.legend([f(markers[i], "k") for i in range(4)], ['tags', 'labels','azure', 'imagenet'],loc=1, framealpha=1)
        #plt.show()
#plt.show()

from pingouin import ancova
import pandas as pd

df = pd.read_csv('tags/covariance.csv')
print(df)
# perform ANCOVA
#covar_results = ancova(data=df, dv='observed', covar='class', between='predicted')
#print(covar_results)

#exit()

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

from createTagsFiles import getTagsFromPredictions
from addGoogleLabels import getLabelsFromPredictions
from addAzureLabels import getAzureTagsFromPredictions
from addGluonLabels import getGluonFromPredictions
import random
for species in ['cane_toad']:
    for source in ['ala_2.', 'flickr', 'twitter', 'reddit', 'inaturalist', 'random']:
        url_and_tags_dict = {}

        for classifier, classifier_results in [['tags', getTagsFromPredictions],
                                               ['labels', getLabelsFromPredictions],
                                               ['azure', getAzureTagsFromPredictions],
                                               ['imagenet', getGluonFromPredictions]]:
            if source=='ala_2.':
                url_and_tags = classifier_results('ala', species)
                url_and_tags = url_and_tags[1::2]
            else:
                url_and_tags = classifier_results(source, species)

            if classifier=='tags':
                random.shuffle(url_and_tags)
                url_and_tags_orig = url_and_tags
                url_and_tags_dict[classifier] = url_and_tags

            else:
                url_and_tags_new = []
                for item1 in url_and_tags_orig:
                    for item2 in url_and_tags:
                        if item1[0]==item2[0]:
                            url_and_tags_new.append(item2)
                            break



                url_and_tags_dict[classifier] = url_and_tags_new

        for classifier, classifier_results in [['tags', getTagsFromPredictions],
                                                   ['labels', getLabelsFromPredictions],
                                                   ['azure', getAzureTagsFromPredictions],
                                                   ['imagenet', getGluonFromPredictions]]:

            url_and_tags = url_and_tags_dict[classifier]

            sample_no = 5

            for sampleIndex in range(sample_no):
                sample = url_and_tags[sampleIndex::sample_no]
                # write new file with image urls and prediction percentages
                with open('predictions/cane_toad_samples/' + source + str(sampleIndex)+'_'+classifier+'.csv', 'w') as myfile:
                    wr = csv.writer(myfile, delimiter=',')
                    wr.writerows(sample)

    for source in ['ala']:

        for classifier, classifier_results in [['tags', getTagsFromPredictions],
                                               ['labels', getLabelsFromPredictions],
                                               ['azure', getAzureTagsFromPredictions],
                                               ['imagenet', getGluonFromPredictions]]:
            url_and_tags = classifier_results(source, species)
            url_and_tags = url_and_tags[::2]

            with open('predictions/cane_toad_samples/' + source+'_'+classifier+'.csv', 'w') as myfile:
                wr = csv.writer(myfile, delimiter=',')
                wr.writerows(url_and_tags)

exit()


# instagram
from instaloader import Instaloader
import datetime
import time
loader = Instaloader()
NUM_POSTS = 10


def get_hashtags_posts(mainTag, maxCount, additionalTag=None):
    posts = loader.get_hashtag_posts(mainTag)
    urls = []
    count = 0
    for post in posts:
        # skip all posts from december/november
        if post.date>datetime.datetime(2021, 2, 25):
            continue
        print(post.date)
        time.sleep(1)
        if not additionalTag or additionalTag in post.caption_hashtags:
            urls.append(post.url)
            count += 1
            if count == maxCount:
                return urls

instagram_urls = get_hashtags_posts('germanwasp', 400)

new_rows=[]
with open('instagram_images/instagram_tag_predictions.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        element=lines
        if len(lines)>0:
            if lines[0]=='url':
                continue
            url = lines[0]

            file_name = url.replace('/','')
            file_name = file_name.replace(':','')

            try:
                file_image = Image.open('instagram_images/'+file_name)
            except:
                print(url)
                continue

            for insta_url in instagram_urls:
                img = urllib.request.urlopen(insta_url)
                img = Image.open(img)
                if file_image==img:
                    element[0]=insta_url
                    print('found')
                    instagram_urls.remove(insta_url)
                    break
        new_rows.append(element)

with open('predictions/german_wasp/instagram_tag_predictions.csv', "w") as csv_file:
    wr = csv.writer(csv_file, delimiter=',')
    wr.writerows(new_rows)

exit()



url_and_tags = []
with open('predictions/instgramCaneToad_all_tag_predictions.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        if len(lines)>0:
            if lines[0]=='url':
                continue
            url = lines[0]
            crop = lines[1]
            tags = lines[2]
            unPre = lines[3]
            crPre = lines[4]

            reid = ''
            note = ''

            # remove quotation marks from each string
            url_and_tags.append([url, crop, tags, unPre, crPre, reid, note])



with open('predictions/reid/instagram_tag_predictions.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    url_and_tags_new = []
    for lines in csv_reader:
        # skip first line
        if lines[0]=='url':
            continue
        crop = lines[1]
        tags = lines[2]
        unPre = float(lines[3])
        crPre = float(lines[4])
        reid = lines[5]
        note = lines[6]
        url = lines[0]

        for index,element in enumerate(url_and_tags):
            if crop != 'NA' and [crop,tags] == element[1:3]:
                element[1] = crop
                element[3] = unPre
                element[4] = crPre
                element[5] = reid
                element[6] = note
                url_and_tags_new.append(element)
            elif crop=='NA' and tags == element[2] and "%.3f %.3f" %(unPre,crPre) == "%.3f %.3f" %(float(element[3]),float(element[4])):
                print('hi')
                element[1] = crop
                element[3] = element[3]
                element[4] = element[4]
                element[5] = reid
                element[6] = note
                url_and_tags_new.append(element)

print(url_and_tags_new)


with open('predictions/reid/instagram_all_tag_predictions.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(url_and_tags_new)