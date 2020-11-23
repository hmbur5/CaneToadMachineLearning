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

# write new file with image urls and prediction percentages
with open('tags/summary.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(['website', 'curve fit', 'number of tags to cover 90% of images','proprtion of images covered by frog and animal','proportion of images with multiple tags', 'proportion of images with any animal tag', 'proportion of images with multiple animal tags', 'proportion of images with animal and human'])

    for source in ['instagramMink']:

        url_and_tags = []
        with open('tags/' + source + '.csv', "r") as csv_file:
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
                # avoid any duplicates
                if [url, newTags] not in url_and_tags:
                    url_and_tags.append([url, newTags])
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
            image_url = element[0]
            img = urllib.request.urlopen(image_url)
            img = Image.open(img)
            if img not in open_images:
                open_images.append(img)
                url_and_tags_new.append(element)
            else:
                print('hi')
        url_and_tags = url_and_tags_new


        # create histogram
        tagsList = []
        for url, tags in url_and_tags:
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

        plt.bar(ticks[0:25], counts[0:25], align='center', color='orange')
        plt.xticks(ticks[0:25], labels[0:25], rotation='vertical')
        plt.title(source)

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
        plt.show()



        # minumum number of tags to cover 90% of images
        print('number of tags covering 90% of images:')
        covered = []
        for index in range(len(url_and_tags)):
            if len(covered)>0.90*len(url_and_tags):
                break
            tag = labels[index]
            for url, tags in url_and_tags:
                if tag in tags and url not in covered:
                    covered.append(url)
        row_to_add.append(index)
        print(index)

        # number of images frog and animal cover
        print('number of images animal and otter cover:')
        covered = []
        for tag in ['Otter', 'Animal']:
            for url, tags in url_and_tags:
                if tag in tags and url not in covered:
                    covered.append(url)
        row_to_add.append("%.4f"%(len(covered)/len(url_and_tags)))
        print(len(covered)/len(url_and_tags))


        # percentage of images with multiple tags:
        #print('proportion of some tags')
        url_and_tags_multiple = []
        url_and_tags_some = []
        for url, tags in url_and_tags:
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
                   'Otter', 'Brown bear', 'Raccoon', 'Squirrel', 'Porcupine']

        # percentage of images with multiple animal tags:
        url_and_tags_multiple_animals = []
        url_and_tags_animal_and_human = []
        url_and_tags_animals = []
        missing = []

        for url, tags in url_and_tags:
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



