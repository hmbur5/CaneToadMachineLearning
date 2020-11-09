from createTagsFiles import getTagsFromPredictions
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import csv

# write new file with image urls and prediction percentages
with open('tags/summary.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(['website', 'curve fit', 'number of tags to cover 90% of images','proprtion of images covered by frog and animal','proportion of images with multiple tags', 'proportion of images with multiple animal tags', 'of images with animals, proportion of images with multiple animal tags'])

    for source in ['instagram', 'twitter', 'flickr', 'ala', 'reddit', 'inaturalist']:
        row_to_add = []

        print(source)
        row_to_add.append(source)

        # creating histogram
        url_and_tags = getTagsFromPredictions(source)
        CtagsList = []
        NtagsList = []
        image_url_list = []
        url_and_tags_new = []
        # remove any images without tags
        for index, element in enumerate(url_and_tags):
            if len(element[1])>0:
                url_and_tags_new.append(url_and_tags[index])
        url_and_tags = url_and_tags_new

        for url,tags,coords,prediction in url_and_tags:
            if prediction>0.95:
                CtagsList+=list(set(tags))
            else:
                NtagsList+=list(set(tags))
            image_url_list.append(url)
        tagsList = CtagsList + NtagsList

        labels, counts = np.unique(tagsList, return_counts=True)
        # sort in descending order
        sorted_indices = np.argsort(-counts)
        counts = counts[sorted_indices]
        labels = labels[sorted_indices]

        # split counts into cane toad and not cane toad
        Ccounts = [0.0]*len(counts)
        Ncounts = [0.0]*len(counts)
        for index, label in enumerate(labels):
            for tag in CtagsList:
                if tag==label:
                    Ccounts[index]+=1
            for tag in NtagsList:
                if tag==label:
                    Ncounts[index]+=1


        #counts = counts[0:25]
        #labels = labels[0:25]
        ticks = range(len(counts))
        # normalise to proportion of total number of images
        #counts = counts / len(image_url_list)
        # normalise so peak is at 1
        Ccounts = Ccounts / max(counts)
        counts = counts / max(counts)

        plt.bar(ticks[0:25], counts[0:25], align='center', color='orange')
        plt.bar(ticks[0:25], Ccounts[0:25], align='center', color='green')
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
            for url, tags, coords, prediction in url_and_tags:
                if tag in tags and url not in covered:
                    covered.append(url)
        row_to_add.append(index)
        print(index)

        # number of images frog and animal cover
        print('number of images frog and animal cover:')
        covered = []
        for tag in ['Frog', 'Animal']:
            for url, tags, coords, prediction in url_and_tags:
                if tag in tags and url not in covered:
                    covered.append(url)
        row_to_add.append(len(covered)/len(url_and_tags))
        print(len(covered)/len(url_and_tags))


        # percentage of images with multiple tags:
        #print('proportion of some tags')
        url_and_tags_multiple = []
        url_and_tags_some = []
        for url, tags, coords, prediction in url_and_tags:
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
                   'Worm', 'Lizard', 'Insect', 'Fish', 'Oyster', 'Tortoise', 'Sea turtle', 'Cat', 'Sheep', 'Snail']

        # percentage of images with multiple animal tags:
        url_and_tags_multiple_animals = []
        url_and_tags_animals = []
        missing = []

        for url, tags, coords, prediction in url_and_tags:
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




        # object size
        #areas = []
        #for url, tags, coords, prediction in url_and_tags:
        #    x1=coords[0]
        #    y1=coords[1]
        #    x2=coords[2]
        #    y2=coords[3]
        #    # provided there is an object
        #    if x1 or x2 or y1 or y2:
        #        area = abs((x1-x2)*(y1-y2))
        #        areas.append(area)
        #print(areas)
        #plt.hist(areas, bins=10)
        #plt.title(source)
        #plt.show()


        wr.writerow(row_to_add)
