from createTagsFiles import getTagsFromPredictions
import urllib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from addGoogleLabels import getLabelsFromPredictions



def filter(source, url_and_tags_comparison, filterSource = 'ala', filter=True, url_and_tags_filtered = [], check_rms = False):
    # get ala images
    # if comparing objects, use getTags, if comparing labels, use getLabels
    # returns same shape list just there are more detailed names in the tags list.
    #url_and_tags = getLabelsFromPredictions(filterSource)
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
    for url, tags, coords, prediction, reid in url_and_tags:
        tagsListALA += list(set(tags))

    no_imagesALA = len(url_and_tags)

    ALAlabels, ALAcounts = np.unique(tagsListALA, return_counts=True)
    # sort in descending order


    # now analyse the given dataset
    url_and_tags = url_and_tags_comparison

    tagsList = []
    for url, tags, coords, prediction, reid in url_and_tags:
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
    print('RMS error from ala')
    print(rms_error)


    # filtering
    if filter:
        '''
        # filtering based on deleting images with tags whose error is above some threshold
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
                url_and_tags_filtered.append([url, tags, coords, prediction, reid])
        '''

        # filtering based on a metric of how the tags in each image compare to ala distribution

        metricDict = {}
        values = []
        url_and_tags_filtered = []
        for url, tags, coords, prediction, reid in url_and_tags:
            value = 0
            for index, count in enumerate(counts):
                if labels[index] in tags:
                    # if there are less of this tag in image set than ala and it appears in this photo, increase value
                    # based on the frequency difference (which would be negative)
                    value -= count
                else:
                    # if the tag doesn't appear
                    value += count
                    pass
            metricDict[url] = value
            values.append(value)


        # printing bars to show how order of metric compares to actual images of cane toads/not cane toads
        orderedUrls = []
        values = np.sort(values)
        for value in values:
            for url, tags, coords, prediction, reid in url_and_tags:
                if metricDict[url] == value:
                    orderedUrls.append([url, reid])
        uniqueOrderedUrls = []
        for item in orderedUrls:
            if item not in uniqueOrderedUrls:
                uniqueOrderedUrls.append(item)




        xVal = 0.05
        global yVal
        global xMax
        from termcolor import colored
        for url, reid in uniqueOrderedUrls:
            if reid=='T' or reid=='PT':
                plt.text(xVal, yVal,
                         '|',
                         color="green")
            else:
                plt.text(xVal, yVal,
                         '|',
                         color="red")
            xVal+=0.07
        plt.text(xVal/2, yVal+0.03,source,color='black')
        plt.ylim(0,yVal)
        if xVal>xMax:
            plt.xlim(0, xVal+0.05)
            xMax = xVal
        yVal+=0.08





        # based on initial rms error (which as shown is proportional to number of desired images), choose threshold
        values = np.sort(values)
        # if rms error = 0.1, we want to cut half of the images
        cut = rms_error/0.2
        print(cut)
        index = int(cut*len(values))
        threshold = values[index]

        url_and_tags_filtered = []
        for url, tags, coords, prediction, reid in url_and_tags:
            if metricDict[url]>threshold:
                url_and_tags_filtered.append([url, tags, coords, prediction, reid])

    if check_rms:
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

        print(len(url_and_tags))
        print(len(url_and_tags_filtered))

        abscounts = []
        for count in counts:
            abscounts.append(-abs(count))

        sorted_indices = np.argsort(abscounts)
        sortedCounts = []
        sortedLabels = []
        for i in sorted_indices:
            sortedCounts.append(counts[i])
            sortedLabels.append(labels[i])

        rms_filtered_error = np.sqrt(np.mean(np.square(sortedCounts)))
        print('RMS filtered error from ala')
        print(rms_filtered_error)

        '''ticks = range(len(sortedCounts))
        plt.clf()
        plt.bar(ticks[0:25], sortedCounts[0:25], align='center')
        plt.xticks(ticks[0:25], sortedLabels[0:25], rotation='vertical')
        plt.title(source)
        plt.ylim([-0.5, 0.5])
        plt.show()'''


    return (url_and_tags_filtered, rms_error, rms_filtered_error)





def filter_plot(stats):
    plt.show()
    plt.plot()
    colors = ['red', 'blue', 'green', 'orange', 'pink']
    color_index = 0
    sources=[]
    for source, prop_CT, rms_error, prop_filtered_CT, rms_filtered_error in stats:
        arrow = plt.arrow(rms_error, prop_CT, rms_filtered_error-rms_error, prop_filtered_CT-prop_CT, color=colors[color_index])
        color_index+=1
        sources.append(source)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                    Line2D([0], [0], color=colors[1], lw=4),
                    Line2D([0], [0], color=colors[2], lw=4),
                    Line2D([0], [0], color=colors[3], lw=4)]
    plt.legend(custom_lines, sources)
    plt.title('Expert verified vs RMS of tag frequency difference from ALA')
    plt.xlabel('RMS of tag frequency difference between image set & ALA')
    plt.ylabel('Fraction of images containing cane toads')
    plt.show()



if __name__ == '__main__':


    stats = []
    global yVal
    global xMax
    xMax = 0
    yVal = 0.05
    plt.figure()

    for source in ['flickr', 'inaturalist', 'twitter', 'reddit', 'instagram_all']:

        print(source)

        # if comparing objects, use getTags, if comparing labels, use getLabels
        # returns same shape list just there are more detailed names in the tags list.
        url_and_tags = getTagsFromPredictions(source)
        random.shuffle(url_and_tags)
        #url_and_tags = getLabelsFromPredictions(source)

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
        prop_CT = len(url_and_tags_canetoad) / len(url_and_tags)
        print(prop_CT)


        url_and_tags_filtered, rms_error, rms_filtered_error = filter(source, url_and_tags, check_rms=True)



        url_and_tags_canetoad = []
        for url, tags, coords, prediction, reid in url_and_tags_filtered:
            if reid == 'T' or reid == 'PT':
                url_and_tags_canetoad.append([url, tags, coords, prediction, reid])
        print('proportion verified cane toad filtered')
        prop_filtered_CT = len(url_and_tags_canetoad) / len(url_and_tags_filtered)
        print(prop_filtered_CT)

        stats.append([source, prop_CT, rms_error, prop_filtered_CT, rms_filtered_error])

    filter_plot(stats)





