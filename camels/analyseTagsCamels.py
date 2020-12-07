#from createTagsFiles import getTagsFromPredictions
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import csv
import urllib.request
from PIL import Image
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
from google.cloud import vision
from google.cloud.vision import types
import io
import math


def colour_distance(RGB1, RGB2):
    R1,G1,B1 = RGB1
    R2,G2,B2 = RGB2
    cR = R1 - R2
    cG = G1 - G2
    cB = B1 - B2
    uR = R1 + R2
    distance = cR * cR * (2 + uR / 256) + cG * cG * 4 + cB * cB * (2 + (255 - uR) / 256)
    return distance

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v




def step(r, g, b, repetitions=1):
    r=r*255
    g=g*255
    b=b*255
    lum = math.sqrt(.241 * r + .691 * g + .068 * b)

    h, s, v = rgb_to_hsv(r, g, b)

    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)

    if h2 % 2 == 1:
        v2 = repetitions - v2
    lum = repetitions - lum

    return (h2, lum, v2)


# write new file with image urls and prediction percentages
with open('tags/summary.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(['website', 'proportion verified camels','curve fit', 'rms error from inat','number of tags to cover 90% of images',
                 'proprtion of images covered by camel and animal','proportion of images with multiple tags',
                 'proportion of images with any animal tag', 'proportion of images with multiple animal tags',
                 'proportion of images with animal and human', 'min colour distance from (168, 107, 50)'])

    url_and_tags_all = []

    for source in ['inaturalist','instagram', 'twitter', 'flickr', 'ala', 'reddit']:


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



        row_to_add = []

        print(source)
        row_to_add.append(source)



        # remove any images without tags
        url_and_tags_new = []
        for index, element in enumerate(url_and_tags):
            if len(element[1])>0:
                url_and_tags_new.append(url_and_tags[index])
        url_and_tags = url_and_tags_new



        # remove duplicate images
        url_and_tags_new = []
        open_images = []
        for index, element in enumerate(url_and_tags):
            try:
                image_url = element[0]
                img = urllib.request.urlopen(image_url)
                img = Image.open(img)
                if img not in open_images:
                    open_images.append(img)
                    url_and_tags_new.append(element)
                else:
                    print('duplicate')
            except urllib.error.HTTPError:
                pass
        url_and_tags = url_and_tags_new


        url_and_tags_all+=url_and_tags


        url_and_tags_camel = []
        for url,tags,hannah in url_and_tags:
            if hannah=='C':
                url_and_tags_camel.append([url, tags, hannah])
        print('proportion verified camel')
        print(len(url_and_tags_camel)/len(url_and_tags))
        row_to_add.append("%.4f"%(len(url_and_tags_camel)/len(url_and_tags)))


        # create histogram
        tagsList = []
        for url, tags, hannah in url_and_tags:
            tagsList+= list(set(tags))
        print(tagsList)
        # create histogram
        labels, counts = np.unique(tagsList, return_counts=True)
        # sort in descending order
        sorted_indices = np.argsort(-counts)
        counts = counts[sorted_indices]
        labels = labels[sorted_indices]


        #counts = counts[0:25]
        #labels = labels[0:25]
        ticks = range(len(counts))
        # normalise to proportion of total number of images
        #counts = counts / len(image_url_list)
        # normalise so peak is at 1
        counts = counts / max(counts)

        plt.clf()
        plt.bar(ticks[0:25], counts[0:25], align='center', color='orange')
        plt.xticks(ticks[0:25], labels[0:25], rotation='vertical')
        plt.gcf().subplots_adjust(bottom=0.35)
        plt.title(source)
        #plt.show()

        # creating exponential fit
        def func(t, b):
            return np.exp(-b*t)
        popt, pcov = curve_fit(func, ticks, counts)
        xx = np.linspace(0, len(ticks[0:25]), 1000)
        yy = func(xx, *popt)/max(counts)
        print('rate of decay:')
        print(popt)
        row_to_add.append("e^( -%.2f * n)"%(popt[0]))
        plt.plot(xx,yy,'b')
        #plt.show()


        # compare distributions
        if source == 'instagram':
            tagsListInstagram = tagsList
            no_imagesInstagram = len(url_and_tags)
        if source == 'twitter':
            tagsListTwitter = tagsList
            no_imagesTwitter = len(url_and_tags)
        if source == 'reddit':
            tagsListReddit = tagsList
            no_imagesReddit = len(url_and_tags)
        if source=='flickr':
            tagsListFlickr = tagsList
            no_imagesFlickr = len(url_and_tags)
        if source=='ala':
            tagsListALA = tagsList
            no_imagesALA = len(url_and_tags)
            row_to_add.append("0")
        if source == 'inaturalist':
            tagsListInat = tagsList
            no_imagesInat = len(url_and_tags)
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
                        Inatcounts[index] +=1

            for index, count in enumerate(counts):
                Inatcounts[index] = Inatcounts[index]/no_imagesInat
                count = count/len(url_and_tags)
                try:
                    counts[index] = count-Inatcounts[index]
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
            plt.title(source)
            plt.ylim([-0.5,0.5])
            #plt.show()

            rms_error  = np.sqrt(np.mean(np.square(sortedCounts)))
            row_to_add.append('%.4f' %(rms_error))
            print('RMS error from inaturalist')
            print(rms_error)

        continue



        # minumum number of tags to cover 90% of images
        print('number of tags covering 90% of images:')
        covered = []
        for index in range(len(url_and_tags)):
            if len(covered)>0.90*len(url_and_tags):
                break
            tag = labels[index]
            for url, tags, hannah in url_and_tags:
                if tag in tags and url not in covered:
                    covered.append(url)
        row_to_add.append(index)
        print(index)

        # number of images frog and animal cover
        print('number of images animal and camel cover:')
        covered = []
        for tag in ['Camel', 'Animal']:
            for url, tags, hannah in url_and_tags:
                if tag in tags and url not in covered:
                    covered.append(url)
        row_to_add.append("%.4f"%(len(covered)/len(url_and_tags)))
        print(len(covered)/len(url_and_tags))


        # percentage of images with multiple tags:
        #print('proportion of some tags')
        url_and_tags_multiple = []
        url_and_tags_some = []
        for url, tags, hannah in url_and_tags:
            if len(tags)>=1:
                url_and_tags_some.append([url, tags])
            if len(tags)>1:
                url_and_tags_multiple.append([url, tags])
        #print(len(url_and_tags_some)/len(url_and_tags))
        #row_to_add.append("%.4f"%(len(url_and_tags_some)/len(url_and_tags)))
        print('proportion of multiple tags')
        print(len(url_and_tags_multiple)/len(url_and_tags))
        row_to_add.append("%.4f"%(len(url_and_tags_multiple)/len(url_and_tags)))


        animals = ['Animal','Frog', 'Crab', 'Dog', 'Moths and butterflies', 'Snake', 'Bird', 'Turtle', 'Scorpion',
                   'Worm', 'Lizard', 'Insect', 'Fish', 'Oyster', 'Tortoise', 'Sea turtle', 'Cat', 'Sheep', 'Snail',
                   'Otter', 'Brown bear', 'Raccoon', 'Squirrel', 'Porcupine', 'Camel', 'Deer', 'Centipede', 'Alpaca',
                   'Goat', 'Eagle']

        # percentage of images with multiple animal tags:
        url_and_tags_multiple_animals = []
        url_and_tags_animal_and_human = []
        url_and_tags_animals = []
        missing = []

        for url, tags, hannah in url_and_tags:
            animalsTags = [x for x in tags if x in animals]
            missing+=[x for x in tags if x not in animals]
            if len(animalsTags)>=1:
                url_and_tags_animals.append([url, animalsTags])
                if 'Person' in tags:
                    url_and_tags_animal_and_human.append([url, animalsTags+['Person']])
            if len(animalsTags)>1:
                url_and_tags_multiple_animals.append([url, animalsTags])

        print('proportion of photos with animals')
        print(len(url_and_tags_animals)/len(url_and_tags))
        row_to_add.append("%.4f" %(len(url_and_tags_animals)/len(url_and_tags)))
        print('proportion of multiple animal tags')
        print(len(url_and_tags_multiple_animals)/len(url_and_tags))
        row_to_add.append("%.4f" %(len(url_and_tags_multiple_animals)/len(url_and_tags)))
        print('proportion of animal and person tags')
        print(len(url_and_tags_animal_and_human) / len(url_and_tags))
        row_to_add.append("%.4f" % (len(url_and_tags_animal_and_human) / len(url_and_tags)))


        print('missing tags')
        print(list(set(missing)))



        '''
        # colours
        # combine whole image set to find dominant colours in that set

        new_im = Image.new('RGB', (100 * len(url_and_tags), 100))
        index = 0
        for url, tags, hannah in url_and_tags:
            img = urllib.request.urlopen(url)
            img = Image.open(img)
            img = img.resize((100, 100))
            new_im.paste(img, (index, 0))
            index += 100
        new_im.show()

        img_byte_arr = io.BytesIO()
        new_im.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()



        plt.clf()
        distances = []
        pix = new_im.load()
        for x in range(0, 100 * len(url_and_tags)):
            for y in range(0, 100):
                distances.append(colour_distance((168, 107, 50), pix[x, y]))
        plt.hist(distances, bins=50, density=True)
        plt.title(source)
        # plt.show()
        y, x, _ = plt.hist(distances, bins=50, density=True)
        row_to_add.append(max(y))'''


        wr.writerow(row_to_add)





# errors of each website
labels = []
countsDict = {}

for source in ['inaturalist','ala','instagram','flickr','twitter','reddit']:
    if source == 'ala':
        tagsList = tagsListALA
        no_images = no_imagesALA
    if source == 'instagram':
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
        tagsList = tagsListInat
        no_images = no_imagesInat


    labels_new, counts = np.unique(tagsList, return_counts=True)
    for label in labels_new:
        labels = np.append(labels, labels_new)
    labels = list(set(labels))


    counts = [0.0] * len(labels)
    for index, label in enumerate(labels):
        for tag in tagsList:
            if tag == label:
                counts[index] += 1

    for index,count in enumerate(counts):
        counts[index] = count/no_images

    for index,label in enumerate(labels):
        try:
            countsDict[label].append([counts[index],source])
        except KeyError:
            countsDict[label] = [[counts[index], source]]


InatComparison = {}
allCounts = []
allLabels = []
for label in countsDict.keys():
    list = countsDict[label]
    for counts, source in list:
        if source=='inaturalist':
            InatComparison[label] = counts
    # if there were no counts of this tag
    if label not in InatComparison:
        InatComparison[label] = 0

    # convert all other counts to errors
    for index, tuple in enumerate(list):
        list[index][0] = tuple[0] - InatComparison[label]

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
plt.xticks(ticks[0:25], sortedLabels[0:25], rotation='vertical')
plt.ylim([-0.5, 0.5])
for source in ['flickr','instagram','twitter','reddit','ala']:
    counts = []
    for label in sortedLabels:
        for tuple in countsDict[label]:
            if tuple[1]==source:
                break
        if tuple[1]!= source:
            counts.append(0)
        else:
            counts.append(tuple[0])

    ticks = range(len(counts))
    plt.scatter(ticks[0:25], counts[0:25])

plt.gcf().subplots_adjust(bottom=0.35)
plt.ylabel('Tag frequency deviation from iNaturalist')
plt.legend(['flickr','instagram','twitter','reddit','ala'])
plt.show()





