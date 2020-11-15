import csv

url_and_tags = []
with open('tags/instgramCaneToad.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        url = lines[0]
        tags = lines[2]
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
        crop = lines[1]
        unPre = lines[3]
        crPre = lines[4]
        reid = lines[5]
        note = lines[6]
        url = lines[0]


        for index,element in enumerate(url_and_tags):
            orig_url = element[0]
            print(orig_url)
            if orig_url == url:
                element[1] = crop
                element[3] = unPre
                element[4] = crPre
                element[5] = reid
                element[6] = note
                url_and_tags_new.append(element)



with open('predictions/reid/instagram_tag_predictions_adjusted.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(url_and_tags_new)