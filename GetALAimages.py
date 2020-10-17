import csv
import matplotlib.pyplot as plt
from skimage import io
import sys
import csv
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

manualConfirm = None

def listOfAlaImageUrls(file_dir):
    '''
    Gets urls of all images from an occurrence search download, based on file given by ALA

    The raw file comes from ALA using a download url in the following form (change search field for different species):
    https://biocache-ws.ala.org.au/ws/occurrences/offline/download*?q=cane%20toad&email=hmbur5%40student.monash.edu&fields=all_image_url
    which sends a link to your email.
    This image url is then put into the form https://images.ala.org.au/store/b/8/a/0/d6ea9ad8-0293-4144-b40e-9087eb400a8b/original
    where the first 4 digits are the reverse of the last 4 digits from the giant 'url'
    The other columns in this file relate to quality test warnings: which are true if it is a warning (this could be used
    to give a value of quality of data)
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


def listOfCheckedImages(file_dir):
    image_url_list = []
    with open(file_dir, 'r', newline='') as f:
        for line in f:
            line = line.replace('\r\n', '')
            image_url_list.append(line)
    return image_url_list

def manualConfirmationOfImages(unchecked_image_urls):
    '''
    Function to open a set of images, and wait for Y or N to split them into two lists
    (one of cane toad training images, one of not cane toads)
    This is necessary as some ALA images are of skeletons, tad poles etc
    As the list is so large, the files are built up iteratively
    :param unchecked_image_urls: list of image urls
    '''

    # create list of all processed images
    processedURLS = []
    try:
        with open('ala image urls/confirmedCaneToads.csv', 'r', newline='') as f:
            for line in f:
                line = line.replace('\r\n','')
                processedURLS.append(line)
        with open('ala image urls/confirmedNotCaneToads.csv', 'r', newline='') as f:
            for line in f:
                line = line.replace('\r\n', '')
                processedURLS.append(line)
    except FileNotFoundError:
        pass

    for image_url in unchecked_image_urls:
        if image_url in processedURLS:
            continue
        # print each image and wait for key press
        image = io.imread(image_url)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title('c for cane toad, n for not cane toad')
        # calls function press when key is pressed
        fig.canvas.mpl_connect('key_press_event', lambda event: press(event, image_url, plt))
        plt.show()


def press(event, image_url, plt):
    sys.stdout.flush()
    if event.key == 'c':
        plt.close()
        with open('ala image urls/confirmedCaneToads.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([[image_url]])
    elif event.key == 'n':
        plt.close()
        with open('ala image urls/confirmedNotCaneToads.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([[image_url]])



def manualConfirmationOfTest(image):
    global manualConfirm
    '''
    Function to open an image, and wait for key press C or N to return True if user decides it is a cane toad, and
    false if not
    This is necessary as some ALA images are of skeletons, tad poles etc
    :param image_url: image urls
    '''
    # print each image and wait for key press
    #image = io.imread(image_url)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title('c for cane toad, n for not cane toad')
    # calls function press when key is pressed
    fig.canvas.mpl_connect('key_press_event', lambda event: pressTest(event, plt))
    plt.show()
    return manualConfirm

def pressTest(event, plt):
    global manualConfirm
    sys.stdout.flush()
    if event.key == 'c':
        plt.close()
        manualConfirm = True
    elif event.key == 'n':
        plt.close()
        manualConfirm = False


file_dir = 'ala image urls/caneToadRawFile.csv'
unchecked_image_urls = listOfAlaImageUrls(file_dir)
manualConfirmationOfImages(unchecked_image_urls)

