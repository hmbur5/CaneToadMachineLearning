import csv

url_and_tags = []
with open('predictions/flickr_tag_predictions.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        url = lines[0]
        crop = lines[1]
        tags = lines[2]
        unPre = lines[3]
        crPre = lines[4]

        # remove quotation marks from each string
        url_and_tags.append([url, crop, tags, unPre, crPre])


with open('flickr photos/reids_notes.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        correction = lines[0]
        note = lines[1]
        url = lines[2]
        prob = lines[3]

        boolean = ''

        if float(prob)<0.95:
            if correction == 'R':
                boolean = 'not cane toad'
            elif correction == 'W':
                boolean = 'cane toad'
        if float(prob)>0.95:
            if correction == 'R':
                boolean = 'cane toad'
            elif correction == 'W':
                boolean = 'not cane toad'

        for index,element in enumerate(url_and_tags):
            orig_url = element[0]
            print(orig_url)
            if orig_url == url:
                print('h')
                url_and_tags[index] = [url, crop, tags, unPre, crPre,boolean,note]


with open('predictions/flickr_tag_predictions_annotated.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(url_and_tags)