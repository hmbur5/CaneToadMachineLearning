from createTagsFiles import getTagsFromPredictions
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
    wr.writerow(['website', 'proportion >90% cane toad probability', 'proportion verified cane toad','curve fit', 'number of tags to cover 90% of images','proprtion of images covered by frog and animal','proportion of images with multiple tags', 'proportion of images with any animal tag', 'proportion of images with multiple animal tags', 'proportion of images with animal and human', 'proportion of verified predator photos', 'colour distance histogram max'])

    for source in ['ala', 'instagram_new', 'twitter', 'flickr', 'reddit', 'inaturalist']:
        row_to_add = []

        print(source)
        row_to_add.append(source)


        url_and_tags = getTagsFromPredictions(source)

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
        url_and_tags = url_and_tags_new


        # check probability
        CtagsList = []
        NtagsList = []
        image_url_list = []
        url_and_tags_canetoad = []

        for url,tags,coords,prediction, reid in url_and_tags:
            if reid=='T' or reid=='PT':
                CtagsList+=list(set(tags))
            else:
                NtagsList+=list(set(tags))
            if prediction>0.90:
                url_and_tags_canetoad.append([url, tags, coords, prediction, reid])
            image_url_list.append(url)
        print('proportion >90% cane toad probability')
        print(len(url_and_tags_canetoad)/len(url_and_tags))
        row_to_add.append("%.4f"%(len(url_and_tags_canetoad)/len(url_and_tags)))


        url_and_tags_canetoad = []
        for url,tags,coords,prediction, reid in url_and_tags:
            if reid=='T' or reid=='PT':
                url_and_tags_canetoad.append([url, tags, coords, prediction, reid])
        print('proportion verified cane toad')
        print(len(url_and_tags_canetoad)/len(url_and_tags))
        row_to_add.append("%.4f"%(len(url_and_tags_canetoad)/len(url_and_tags)))

        # create histogram
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
        #plt.show()




        # compare distributions
        if source == 'twitter':
            tagsListTwitter = tagsList
            no_imagesTwitter = len(url_and_tags)
        if source == 'reddit':
            tagsListReddit = tagsList
            no_imagesReddit = len(url_and_tags)
        if source == 'inaturalist':
            tagsListInaturalist = tagsList
            no_imagesInaturalist = len(url_and_tags)
        if source=='flickr':
            tagsListFlickr = tagsList
            no_imagesFlickr = len(url_and_tags)
        if source=='ala':
            tagsListALA = tagsList
            no_imagesALA = len(url_and_tags)
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
                        ALAcounts[index] +=1

            for index, count in enumerate(counts):
                ALAcounts[index] = ALAcounts[index]/no_imagesALA
                count = count/len(url_and_tags)
                try:
                    counts[index] = count-ALAcounts[index]
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

        continue

        # minumum number of tags to cover 90% of images
        print('number of tags covering 90% of images:')
        covered = []
        for index in range(len(url_and_tags)):
            if len(covered)>0.90*len(url_and_tags):
                break
            tag = labels[index]
            for url, tags, coords, prediction, reid in url_and_tags:
                if tag in tags and url not in covered:
                    covered.append(url)
        row_to_add.append(index)
        print(index)

        # number of images frog and animal cover
        print('number of images frog and animal cover:')
        covered = []
        for tag in ['Frog', 'Animal']:
            for url, tags, coords, prediction, reid in url_and_tags:
                if tag in tags and url not in covered:
                    covered.append(url)
        row_to_add.append("%.4f"%(len(covered)/len(url_and_tags)))
        print(len(covered)/len(url_and_tags))


        # percentage of images with multiple tags:
        #print('proportion of some tags')
        url_and_tags_multiple = []
        url_and_tags_some = []
        for url, tags, coords, prediction, reid in url_and_tags:
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
        url_and_tags_animal_and_human = []
        url_and_tags_animals = []
        missing = []

        for url, tags, coords, prediction, reid in url_and_tags:
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


        # predator and prey photos
        pred = 0
        for url, tags, coords, prediction, reid in url_and_tags:
            if reid=='PT':
                pred+=1
        print('proportion of verified predator photos')
        print(pred/len(url_and_tags))
        row_to_add.append("%.4f" % (pred/ len(url_and_tags)))

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



        # dominant colours
        '''imageSet_colours = []
        for url, tags, coords, prediction, reid in url_and_tags:

            source = types.ImageSource(image_uri=url)
            image = types.Image(source=source)
            response = client.image_properties(image=image)
            # Performs label detection on the image file
            props = response.image_properties_annotation

            colours = []
            for color in props.dominant_colors.colors:
                dict = {}
                dict['fraction'] = color.pixel_fraction
                dict['red'] = color.color.red
                dict['green'] = color.color.green
                dict['blue'] = color.color.blue
                colours.append(dict)
            print(colours)
            imageSet_colours+=colours
        #print(imageSet_colours)

        size = []
        label_rgb = []
        for color in imageSet_colours:
            size.append(color['fraction'])
            label_rgb.append((color['red']/255,color['green']/255,color['blue']/255))

        label_rgb_new = sorted(label_rgb, key=lambda rgb: step(*rgb))

        print(len(size))
        print(len(label_rgb_new))
        print(len(label_rgb))

        size_new = [0]*len(size)
        for index, element in enumerate(label_rgb):
            for index2, element2 in enumerate(label_rgb_new):
                if element==element2:
                    size_new[index2]=size[index]
        size = size_new/np.sum(size_new)*100
        #size = size[np.argsort(label_hex)]
        #label_rgb_new = []
        #for i in np.argsort(label_hex):
        #    label_rgb_new.append(label_rgb[i])

        plt.clf()
        plt.pie(size, colors=label_rgb)
        plt.title(source)
        plt.show()'''



        # combine whole image set to find dominant colours in that set

        new_im = Image.new('RGB', (100*len(url_and_tags), 100))
        index = 0
        for url, tags, coords, prediction, reid in url_and_tags:
            img = urllib.request.urlopen(url)
            img = Image.open(img)
            img = img.resize((100,100))
            new_im.paste(img, (index, 0))
            index+=100
        new_im.show()

        img_byte_arr = io.BytesIO()
        new_im.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        image = types.Image(content=img_byte_arr)
        response = client.image_properties(image=image)
        # Performs label detection on the image file
        props = response.image_properties_annotation

        colours = []
        for color in props.dominant_colors.colors:
            dict = {}
            dict['fraction'] = color.pixel_fraction
            dict['red'] = color.color.red
            dict['green'] = color.color.green
            dict['blue'] = color.color.blue
            colours.append(dict)
        print(colours)

        sizes = []
        labels = []
        for color in colours:
            sizes.append(color['fraction']*100)
            labels.append((color['red']/255,color['green']/255,color['blue']/255))

        plt.clf()
        plt.pie(sizes, colors=labels)
        plt.title(source)
        #plt.show()

        distances = []
        pix = new_im.load()
        for x in range(0, 100*len(url_and_tags)):
            for y in range(0,100):
                distances.append(colour_distance((100, 82, 60), pix[x,y]))
        plt.hist(distances, bins=50,density=True)
        plt.title(source)
        #plt.show()
        y, x, _ = plt.hist(distances, bins=50,density=True)
        row_to_add.append(max(y))


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
                    if prediction<thresh:
                        false_neg.append(url)
                else:
                    negatives.append(url)
                    if prediction>thresh:
                        false_pos.append(url)
            false_pos_thresh.append(len(false_pos)/len(negatives))
            false_neg_thresh.append(len(false_neg)/len(positives))

            # creating a set of images that are above the given threshold
            above_thresh_set = []
            verified = []
            for url, tags, coords, prediction, reid in url_and_tags:
                if prediction>thresh:
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
                if reid=='T' or reid=='PT':
                    verified.append(url)
                # tags for histogram
                tagsList += list(set(tags))


            if len(above_thresh_set)>0:
                thresholds_valid.append(thresh)
                verified_thresh.append(len(verified)/len(above_thresh_set))
                percentage_frog_animal_thresh.append(len(covered)/len(above_thresh_set))

                # get exponential
                labels, counts = np.unique(tagsList, return_counts=True)
                sorted_indices = np.argsort(-counts)
                counts = counts[sorted_indices]
                counts = counts / max(counts)
                ticks = range(len(counts))
                popt, pcov = curve_fit(func, ticks, counts)
                exponents_thresh.append(popt[0])

        plt.plot(thresholds, false_pos_thresh)
        plt.plot(thresholds, false_neg_thresh)
        plt.title(source)
        plt.legend(['false positives', 'false negatives'])
        plt.xlabel('Probability threshold')
        plt.ylabel('Proportion of wrongly classified images')
        #plt.show()

        plt.plot(thresholds_valid, percentage_frog_animal_thresh)
        plt.plot(thresholds_valid, exponents_thresh)
        plt.plot(thresholds_valid, verified_thresh)
        plt.title(source)
        plt.legend(['percentage frog and animal covered', 'rate of decay', 'percentage verified as cane toad'])
        plt.xlabel('Probability threshold')
        plt.ylabel('Proportion of images out of images above the threshold')
        #plt.show()








# errors of each website
labels = []
countsDict = {}

for source in ['ala','instagram_new','flickr','twitter','reddit','inaturalist']:
    if source == 'ala':
        tagsList = tagsListALA
        no_images = no_imagesALA
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


alaComparison = dict()
allCounts = []
allLabels = []
for label in countsDict.keys():
    list = countsDict[label]
    for counts, source in list:
        if source=='ala':
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
plt.xticks(ticks[0:25], sortedLabels[0:25], rotation='vertical')
plt.ylim([-0.5, 0.5])
for source in ['flickr','instagram_new','twitter','reddit','inaturalist']:
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

plt.legend(['flickr','instagram_new','twitter','reddit','inaturalist'])
plt.show()
