import csv


def listOfAlaImageUrls(file_dir):
    '''
    Gets urls of all images from an occurrence search download, based on file given by ALA

    The raw file comes from ALA using a download url in the following form (change search field for different species):
    https://biocache-ws.ala.org.au/ws/occurrences/offline/download*?q=cane%20toad&email=hmbur5%40student.monash.edu&fields=all_image_url
    which sends a link to your email.
    This image url is then put into the form https://images.ala.org.au/store/b/8/a/0/d6ea9ad8-0293-4144-b40e-9087eb400a8b/original
    where the first 4 digits are the reverse of the last 4 digits from the giant 'url'
    :param file_dir: directory of raw csv file downloaded from ALA
    :return: list of urls
    '''


    url_id_list = []
    with open(file_dir, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
            url_id_list.append(lines[0])


    # for entries that correspond to an image, put in URL form for immediate download
    url_list = []
    for url_id in url_id_list:
        if url_id != '' and 'image' not in url_id:
            url_string = 'https://images.ala.org.au/store/'
            # add last 4 digits in reverse order
            for i in range(1,5):
                url_string += str(url_id[-i])
                url_string += '/'
            url_string += url_id
            url_string += '/original'
            url_list.append(url_string)


    return url_list


