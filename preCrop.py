import os, ssl
import math
import io
import urllib
from PIL import Image
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/hannahburke/Documents/CaneToadMachineLearning/keys.json"
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from io import BytesIO
# Instantiates a client
client = vision.ImageAnnotatorClient()


def cropImage(url, return_coords=False):
    '''
    creates a list of images of the regions of the given image that are identified to
    be a frog or animal based on google cloud
    :param url: image url
    :param return_coords: boolean if coordinates of cropped images are desired
    :return: crops: list of cropped images (a tuple of image and coordinates if desired, coordanites in the form: x1, y1, x2, y2, aspect_ratio as floats)
    :return: tagNames: list of strings corresponding to tag names found in each image url
    '''

    source = types.ImageSource(image_uri=url)

    image = types.Image(source=source)

    # Performs label detection on the image file
    response = client.object_localization(image=image)

    crops = []
    tagNames = []

    for tag in response.localized_object_annotations:
        # some crops that might be cane toads or other frogs
        if tag.name in ['Frog','Animal','Insect']:
            vertices= tag.bounding_poly.normalized_vertices
            im = urllib.request.urlopen(url)
            im = Image.open(im)

            im2 = im.crop([vertices[0].x*im.size[0], vertices[0].y*im.size[1],
                           vertices[2].x*im.size[0], vertices[2].y*im.size[1]])

            #im2.show()

            # if cropped image too large for training/prediction, scaled down
            height, width = im2.size
            # convert to byte object to get size
            imgByteArr = BytesIO()
            im2.save(imgByteArr, format='PNG')
            imgByteArr = imgByteArr.getvalue()
            currentSize = len(imgByteArr)
            if currentSize > 10000000:
                # resize
                maxDim = math.floor(max([width, height]) * 4194304 / currentSize / 4)
                im2.thumbnail((maxDim, maxDim))

            if return_coords:
                crops.append((im2,[vertices[0].x, vertices[0].y,
                           vertices[2].x, vertices[2].y, im.size[0]/im.size[1]]))
            else:
                crops.append(im2)

        else:
            print(tag.name)
        tagNames.append(tag.name)

    return(crops, tagNames)





def caneToadTags():
    import GetALAimages
    import numpy as np
    import matplotlib.pyplot as plt
    file_dir = 'ala image urls/caneToadRawFile.csv'
    image_url_list = GetALAimages.listOfAlaImageUrls(file_dir)
    tagsList = []
    for image_url in image_url_list:
        crops, tags = cropImage(image_url)
        tagsList+=tags
    labels, counts = np.unique(tagsList, return_counts=True)
    sorted_indices = np.argsort(-counts)
    ticks = range(len(counts[sorted_indices]))
    plt.bar(ticks, counts[sorted_indices], align='center')
    plt.xticks(ticks, labels[sorted_indices], rotation='vertical')
    plt.show()


if __name__ == "__main__":
    caneToadTags()


