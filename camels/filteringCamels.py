import urllib.request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

for source in ['inaturalist','ala', 'flickr','instagram', 'twitter', 'flickr', 'reddit']:
    row_to_add = []

    print(source)
    row_to_add.append(source)

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

    url_and_tags_canetoad = []
    for url, tags, hannah in url_and_tags:
        if hannah == 'C':
            url_and_tags_canetoad.append([url, tags, hannah])
    print('proportion verified camel')
    print(len(url_and_tags_canetoad) / len(url_and_tags))


    tagsList = []
    for url, tags, hannah in url_and_tags:
        tagsList += list(set(tags))

    if source == 'inaturalist':
        tagsListInat = tagsList
        no_imagesInat = len(url_and_tags)
        row_to_add.append("0")
    else:
        Inatlabels, Inatcounts = np.unique(tagsListInat, return_counts=True)
        # sort in descending order

        labels, counts = np.unique(tagsList, return_counts=True)
        for label in Inatlabels:
            labels = np.append(labels, label)
        labels = list(set(labels))

        counts = [0.0] * len(labels)
        Inatcounts = [0.0] * len(labels)
        for index, label in enumerate(labels):
            for tag in tagsList:
                if tag == label:
                    counts[index] += 1
            for tag in tagsListInat:
                if tag == label:
                    Inatcounts[index] += 1

        for index, count in enumerate(counts):
            Inatcounts[index] = Inatcounts[index] / no_imagesInat
            count = count / len(url_and_tags)
            try:
                counts[index] = count - Inatcounts[index]
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
        for url, tags, hannah in url_and_tags:
            for tag in below:
                if tag in tags:
                    url_and_tags_filtered.append([url, tags, hannah])
                    continue
            for tag in above:
                if tag in tags:
                    continue
            url_and_tags_filtered.append([url, tags, hannah])


        url_and_tags_canetoad = []
        for url, tags, hannah in url_and_tags_filtered:
            if hannah == 'C':
                url_and_tags_canetoad.append([url, tags, hannah])
        print('proportion verified camel filtered')
        print(len(url_and_tags_canetoad) / len(url_and_tags_filtered))





        tagsList = []
        for url, tags, hannah in url_and_tags_filtered:
            tagsList += list(set(tags))

        labels, counts = np.unique(tagsList, return_counts=True)
        for label in Inatlabels:
            labels = np.append(labels, label)
        labels = list(set(labels))

        counts = [0.0] * len(labels)
        Inatcounts = [0.0] * len(labels)
        for index, label in enumerate(labels):
            for tag in tagsList:
                if tag == label:
                    counts[index] += 1
            for tag in tagsListInat:
                if tag == label:
                    Inatcounts[index] += 1

        for index, count in enumerate(counts):
            Inatcounts[index] = Inatcounts[index] / no_imagesInat
            count = count / len(url_and_tags_filtered)
            try:
                counts[index] = count - Inatcounts[index]
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
        row_to_add.append('%.4f' % (rms_error))
        print('RMS filtered error from inaturalist')
        print(rms_error)

