import csv
import urllib.request
from PIL import Image
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

url_and_tags = []
with open('tags/instgramCaneToad.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        url = lines[0]
        tags = lines[1]
        unPre = ''
        crPre = ''
        crop = ''

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
        unPre = lines[3]
        crPre = lines[4]
        reid = lines[5]
        note = lines[6]
        url = lines[0]

        for index,element in enumerate(url_and_tags):
            if tags == element[2]:
                orig_url = element[0]
                orig_img = urllib.request.urlopen(orig_url)
                orig_img = Image.open(orig_img)
                img = urllib.request.urlopen(url)
                img = Image.open(img)
                if orig_img == img:
                    print('image')
                    element[1] = crop
                    element[3] = unPre
                    element[4] = crPre
                    element[5] = reid
                    element[6] = note
                    url_and_tags_new.append(element)

print(url_and_tags_new)


with open('predictions/reid/instagram_tag_predictions_adjusted.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(url_and_tags_new)