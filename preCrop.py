import os, ssl
import io
import urllib
from PIL import Image
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/hannahburke/Documents/CaneToadMachineLearning/keys.json"
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
# Instantiates a client
client = vision.ImageAnnotatorClient()


def cropImage(url):
    '''
    returns a list of images of the regions of the given image that are identified to be a frog or animal based on google cloud
    :param url: image url
    :return:
    '''

    source = types.ImageSource(image_uri=url)

    image = types.Image(source=source)

    # Performs label detection on the image file
    response = client.object_localization(image=image)

    crops = []

    for tag in response.localized_object_annotations:
        # some crops that might be cane toads
        if tag.name in ['Frog', 'Animal', 'Insect', 'Lizard', 'Snake', 'Moths and butterflies', 'Turtle']:
            vertices= tag.bounding_poly.normalized_vertices
            im = urllib.request.urlopen(url)
            im = Image.open(im)

            im2 = im.crop([vertices[0].x*im.size[0], vertices[0].y*im.size[1],
                           vertices[2].x*im.size[0], vertices[2].y*im.size[1]])

            #im2.show()
            crops.append(im2)

        else:
            print(tag.name)

    return(crops)




#crops = cropImage('https://live.staticflickr.com/8444/7994805572_42f1d484b0_c.jpg')
#for image in crops:
#    image.show()



