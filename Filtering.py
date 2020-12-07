from createTagsFiles import getTagsFromPredictions
import urllib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

for source in ['ala', 'flickr','instagram_new', 'twitter', 'flickr', 'reddit', 'inaturalist']:
    row_to_add = []

    print(source)
    row_to_add.append(source)


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

    url_and_tags_canetoad = []
    for url, tags, coords, prediction, reid in url_and_tags:
        if reid == 'T' or reid == 'PT':
            url_and_tags_canetoad.append([url, tags, coords, prediction, reid])
    print('proportion verified cane toad')
    print(len(url_and_tags_canetoad) / len(url_and_tags))


    tagsList = []
    for url, tags, coords, prediction, reid in url_and_tags:
        tagsList += list(set(tags))

    if source == 'ala':
        tagsListALA = tagsList
        no_imagesALA = len(url_and_tags)
        row_to_add.append("0")
    else:
        ALAlabels, ALAcounts = np.unique(tagsListALA, return_counts=True)
        # sort in descending order

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
        for url,tags,coords,prediction, reid in url_and_tags:
            for tag in below:
                if tag in tags:
                    url_and_tags_filtered.append([url, tags, coords, prediction, reid])
                    continue
            for tag in above:
                if tag in tags:
                    continue
            url_and_tags_filtered.append([url, tags, coords, prediction, reid])


        url_and_tags_canetoad = []
        for url, tags, coords, prediction, reid in url_and_tags_filtered:
            if reid == 'T' or reid == 'PT':
                url_and_tags_canetoad.append([url, tags, coords, prediction, reid])
        print('proportion verified cane toad filtered')
        print(len(url_and_tags_canetoad) / len(url_and_tags_filtered))





        tagsList = []
        for url, tags, coords, prediction, reid in url_and_tags_filtered:
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
        row_to_add.append('%.4f' % (rms_error))
        print('RMS filtered error from ala')
        print(rms_error)

        ticks = range(len(sortedCounts))
        plt.clf()
        plt.bar(ticks[0:25], sortedCounts[0:25], align='center')
        plt.xticks(ticks[0:25], sortedLabels[0:25], rotation='vertical')
        plt.title(source)
        plt.ylim([-0.5, 0.5])
        #plt.show()
