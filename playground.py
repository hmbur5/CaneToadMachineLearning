import csv
import urllib.request
from PIL import Image
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

from createTagsFiles import getTagsFromPredictions
from addGoogleLabels import getLabelsFromPredictions
from addAzureLabels import getAzureTagsFromPredictions
from addGluonLabels import getGluonFromPredictions
import random
for species in ['cane_toad']:
    for classifier in ['tags','labels','azure','imagenet']:
        for source in ['ala_2.', 'flickr', 'twitter', 'reddit', 'inaturalist', 'random']:

            for classifier, classifier_results in [['tags', getTagsFromPredictions],
                                                   ['labels', getLabelsFromPredictions],
                                                   ['azure', getAzureTagsFromPredictions],
                                                   ['imagenet', getGluonFromPredictions]]:
                if source=='ala_2.':
                    url_and_tags = classifier_results('ala', species)
                    url_and_tags = url_and_tags[1::2]

                else:
                    url_and_tags = classifier_results(source, species)
                random.shuffle(url_and_tags)


                sample_no = 5

                for sampleIndex in range(sample_no):
                    sample = url_and_tags[sampleIndex::sample_no]
                    # write new file with image urls and prediction percentages
                    with open('predictions/cane_toad_samples/' + source + str(sampleIndex)+'_'+classifier+'.csv', 'w') as myfile:
                        wr = csv.writer(myfile, delimiter=',')
                        wr.writerows(sample)

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