import urllib
import urllib.request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


def getTagsFromFile(source):
    url_and_tags = []
    with open('tags/verified/' + source + '.csv', "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
            url = lines[0]
            tags = lines[1]
            tags = tags[1:-1]
            if len(tags) == 0:
                newTags = []
            else:
                tags = list(tags.split(", "))  # convert back to list
                # remove quotation marks from each string
                newTags = []
                for tag in tags:
                    newTags.append(tag[1:-1])
            hannah = lines[3]
            # avoid any duplicates
            if [url, newTags, hannah] not in url_and_tags:
                url_and_tags.append([url, newTags, hannah])

            # just use first 250 images
            if len(url_and_tags) >= 250:
                break
    return url_and_tags


def filter(url_and_tags_comparison, filterSource = 'inaturalist', check_rms = False):
    # get ala images
    url_and_tags = getTagsFromFile(filterSource)

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
            pass
            #print(image_url)
    url_and_tags = url_and_tags_new



    tagsListALA = []
    for url,tags,hannah in url_and_tags:
        tagsListALA += list(set(tags))

    no_imagesALA = len(url_and_tags)

    ALAlabels, ALAcounts = np.unique(tagsListALA, return_counts=True)
    # sort in descending order


    # now analyse the given dataset
    url_and_tags = url_and_tags_comparison

    tagsList = []
    for url,tags,hannah in url_and_tags:
        tagsList += list(set(tags))

    labels, counts = np.unique(tagsList, return_counts=True)
    for label in ALAlabels:
        labels = np.append(labels, label)
    labels = list(set(labels))


    counts = [0.0] * len(labels)
    ALAcounts = [0.0] * len(labels)
    for index, label in enumerate(labels):
        for tag in tagsList:
            if tag == label:
                counts[index] += 1
        for tag in tagsListALA:
            if tag == label:
                ALAcounts[index] += 1

    for index, count in enumerate(counts):
        ALAcounts[index] = ALAcounts[index] / no_imagesALA
        count = count / len(url_and_tags)
        try:
            counts[index] = count - ALAcounts[index]
        except IndexError:
            counts[index] = count

    threshold = 0.05
    above = []
    below = []

    for index, count in enumerate(counts):
        if count >threshold:
            above.append(labels[index])
        if count <-1*threshold:
            below.append(labels[index])


    print(above)
    print(below)

    url_and_tags_filtered = []
    for url,tags,hannah in url_and_tags:
        filter = False
        for tag in above:
            if tag in tags:
                filter = True
                break
        for tag in below:
            if tag in tags:
                filter = False
                break
        if not filter:
            url_and_tags_filtered.append([url,tags,hannah])


    if check_rms:
        tagsList = []
        for url,tags,hannah in url_and_tags_filtered:
            tagsList += list(set(tags))

        labels, counts = np.unique(tagsList, return_counts=True)
        for label in ALAlabels:
            labels = np.append(labels, label)
        labels = list(set(labels))

        counts = [0.0] * len(labels)
        ALAcounts = [0.0] * len(labels)
        for index, label in enumerate(labels):
            for tag in tagsList:
                if tag == label:
                    counts[index] += 1
            for tag in tagsListALA:
                if tag == label:
                    ALAcounts[index] += 1

        for index, count in enumerate(counts):
            ALAcounts[index] = ALAcounts[index] / no_imagesALA
            count = count / len(url_and_tags_filtered)
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
        print('RMS filtered error from inaturalist')
        print(rms_error)


    return url_and_tags_filtered


if __name__ == '__main__':
    for source in ['instagram', 'twitter', 'flickr', 'reddit', 'ala']:

        print(source)


        url_and_tags = getTagsFromFile(source)

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
                #print(image_url)
                pass
        url_and_tags = url_and_tags_new


        url_and_tags_camel = []
        for url,tags,hannah in url_and_tags:
            if hannah == 'C':
                url_and_tags_camel.append([url,tags,hannah])
        print('proportion verified camel')
        print(len(url_and_tags_camel) / len(url_and_tags))


        url_and_tags_filtered = filter(url_and_tags, check_rms=True)



        url_and_tags_camel = []
        for url,tags,hannah in url_and_tags_filtered:
            if hannah == 'C':
                url_and_tags_camel.append([url,tags,hannah])
        print('proportion verified camel filtered')
        print(len(url_and_tags_camel) / len(url_and_tags_filtered))






