import csv


# the raw file comes from ALA using the download url:
# https://biocache-ws.ala.org.au/ws/occurrences/offline/download*?q=cane%20toad&email=hmbur5%40student.monash.edu&fields=all_image_url
# which sends a link to your email
# this image url is then put into the form https://images.ala.org.au/store/b/8/a/0/d6ea9ad8-0293-4144-b40e-9087eb400a8b/original
# where the first 4 digits are the reverse of the last 4 digits from the giant 'url'

urls = []
with open('ala image urls/caneToadRawFile.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        urls.append(lines[0])

# delete entries that do not correspond to an image
urlsTemp = []
for URL in urls:
    if URL != '' and URL!='image _ url':
        urlsTemp.append(URL)
urls = urlsTemp
print(urls)
