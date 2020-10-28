from createTagsFiles import getTagsFromFile
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import csv

# write new file with image urls and prediction percentages
with open('tags/summary.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(['website', 'curve fit', 'number of tags >5% of images', 'proportion of images with tags','proportion of images with multiple tags', 'proportion of images with multiple animal tags', 'of images with animals, proportion of images with multiple animal tags'])

    for source in ['flickr']:
        row_to_add = []

        print(source)
        row_to_add.append(source)

        # creating histogram
        url_and_tags = getTagsFromFile(source)
        tagsList = []
        image_url_list = []

        for url,tags,coords in url_and_tags:
            tagsList+=list(set(tags))
            image_url_list.append(url)

        labels, counts = np.unique(tagsList, return_counts=True)
        # crop to first 15 tags
        sorted_indices = np.argsort(-counts)
        counts = counts[sorted_indices]
        labels = labels[sorted_indices]
        #counts = counts[0:25]
        #labels = labels[0:25]
        ticks = range(len(counts))
        # normalise to proportion of total number of images
        counts = counts / len(image_url_list)
        plt.bar(ticks[0:25], counts[0:25], align='center')
        plt.xticks(ticks[0:25], labels[0:25], rotation='vertical')
        plt.title(source)

        # creating exponential fit
        def func(t, a, b):
            return a*np.exp(-b*t)
        popt, pcov = curve_fit(func, ticks, counts)
        xx = np.linspace(0, len(ticks[0:25]), 1000)
        yy = func(xx, *popt)
        print('rate of decay:')
        print(popt)
        row_to_add.append("%.2f * e^( -%.2f * n)"%(popt[0],popt[1]))
        plt.plot(xx,yy,'r')

        # number at which counts<5%
        print('number of tags >5% of images:')
        for index,i in enumerate(counts):
            if i<0.05:
                print(index)
                row_to_add.append(index)
                break


        # percentage of images with multiple tags:
        print('proportion of some tags')
        url_and_tags_multiple = []
        url_and_tags_some = []
        for url, tags,coords in url_and_tags:
            if len(tags)>=1:
                url_and_tags_some.append([url, tags])
            if len(tags)>1:
                url_and_tags_multiple.append([url, tags])
        print(len(url_and_tags_some)/len(url_and_tags))
        row_to_add.append("%.4f"%(len(url_and_tags_some)/len(url_and_tags)))
        print('proportion of multiple tags')
        print(len(url_and_tags_multiple)/len(url_and_tags))
        row_to_add.append("%.4f"%(len(url_and_tags_multiple)/len(url_and_tags)))


        animals = ['Animal','Frog', 'Crab', 'Dog', 'Moths and butterflies', 'Snake', 'Bird', 'Turtle', 'Scorpion',
                   'Worm', 'Lizard', 'Insect', 'Fish', 'Oyster', 'Tortoise', 'Sea turtle', 'Cat', 'Sheep', 'Snail']

        # percentage of images with multiple animal tags:
        url_and_tags_multiple_animals = []
        url_and_tags_animals = []
        missing = []

        for url, tags, coords in url_and_tags:
            animalsTags = [x for x in tags if x in animals]
            missing+=[x for x in tags if x not in animals]
            if len(animalsTags)>=1:
                url_and_tags_animals.append([url, animalsTags])
            if len(animalsTags)>1:
                url_and_tags_multiple_animals.append([url, animalsTags])

        print('proportion of multiple animal tags')
        print(len(url_and_tags_multiple_animals)/len(url_and_tags))
        row_to_add.append("%.4f" %(len(url_and_tags_multiple_animals)/len(url_and_tags)))
        print('proportion of multiple animal tags out of photos with animals')
        print(len(url_and_tags_multiple_animals)/len(url_and_tags_animals))
        row_to_add.append("%.4f" %(len(url_and_tags_multiple_animals)/len(url_and_tags_animals)))

        #print('missing tags')
        #print(list(set(missing)))




        plt.show()

        # object size
        areas = []
        for url, tags, coords in url_and_tags:
            x1=coords[0]
            y1=coords[1]
            x2=coords[2]
            y2=coords[3]
            # provided there is an object
            if x1 or x2 or y1 or y2:
                area = abs((x1-x2)*(y1-y2))
                areas.append(area)
        print(areas)
        plt.hist(areas, bins=10)
        plt.title(source)
        plt.show()


        wr.writerow(row_to_add)