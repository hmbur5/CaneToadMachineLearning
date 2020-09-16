from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateBatch, ImageUrlCreateEntry, ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import GetALAimages
import csv
import time
import numpy as np
from PIL import Image
import urllib
from io import BytesIO
import math




def addImages(image_url_list, tag_name):

    # find tag to add images to
    for tag in trainer.get_tags(project.id):
        if tag.name == tag_name:
            break
    if tag.name!= tag_name:
        raise Exception('Tag name does not exist')


    # going through a small portion of url list as can only upload 64 at a time
    for batch_number in range(math.ceil(len(image_url_list)/64)):
        print(batch_number)

        image_list = []
        endIndex = (batch_number+1)*64
        if endIndex > len(image_url_list):
            endIndex = len(image_url_list)
        for url in image_url_list[batch_number*64: endIndex]:
            image_list.append(ImageUrlCreateEntry(url=url, tag_ids=[tag.id]))

        # check that there are some images to upload, then upload these
        if len(image_list) > 0:
            upload_result = trainer.create_images_from_urls(project.id, ImageUrlCreateBatch(images=image_list))

            # gives error when there is a duplicate (which is possible with the ALA data) so code below is to ignore
            # duplicate error.
            while not upload_result.is_batch_successful:
                image_list = []
                for image in upload_result.images:
                    if image.status == 'OK':
                        # add to new new image_list with corresponding url and tag
                        image_list.append(ImageUrlCreateEntry(url=image.source_url, tag_ids=[tag.id]))
                    elif image.status != 'OKDuplicate':
                        print('resizing')
                        try:
                            # if images are too large for prediction (must be <4mb), use python to scale down
                            img = urllib.request.urlopen(image.source_url)
                            img = Image.open(img)

                            # resize
                            height, width = img.size
                            currentSize = len(img.fp.read())
                            # these numbers could probably be better
                            maxDim = math.floor(max([width, height]) * 4194304 / currentSize / 4)
                            img.thumbnail((maxDim, maxDim))

                            # convert back to byte object for classify_image
                            imgByteArr = BytesIO()
                            img.save(imgByteArr, format='PNG')
                            imgByteArr = imgByteArr.getvalue()

                            image_file_list = [ImageFileCreateEntry(name=image.source_url, contents=imgByteArr,
                                                                    tag_ids=[tag.id])]

                            trainer.create_images_from_files(project.id, ImageUrlCreateBatch(images=image_file_list))

                        # if image actually doesn't exist
                        except OSError:
                            print(image.source_url)
                if len(image_list)>0:
                    upload_result = trainer.create_images_from_urls(project.id, ImageFileCreateBatch(images=image_list))
                else:
                    break




def train(iteration_name):
    # iteration name must be changed each iteration to publish
    publish_iteration_name = iteration_name

    print("Training...")
    iteration = trainer.train_project(project.id)
    while (iteration.status != "Completed"):
        iteration = trainer.get_iteration(project.id, iteration.id)
        print("Training status: " + iteration.status)
        time.sleep(1)

    # The iteration is now trained. Publish it to the project endpoint

    trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
    print("Done!")


def predict(image_url, iteration_name):
    print(iteration_name)

    # get prediction percentages
    try:
        results = predictor.classify_image_url(project.id, iteration_name, image_url)

    # not sure what exception should be as custom vision gives a strange error.
    except:
        print('resizing')
        try:
            # if images are too large for prediction (must be <4mb), use python to scale down
            img = urllib.request.urlopen(image_url)
            img = Image.open(img)

            # resize
            height,width = img.size
            currentSize = len(img.fp.read())
            # these numbers could probably be better
            maxDim = math.floor(max([width,height]) * 4194304 / currentSize / 3)
            img.thumbnail((maxDim, maxDim))


            # convert back to byte object for classify_image
            imgByteArr = BytesIO()
            img.save(imgByteArr, format='PNG')
            imgByteArr = imgByteArr.getvalue()

            results = predictor.classify_image(project.id, iteration_name, imgByteArr)

        # if image actually doesn't exist
        except OSError:
            pass

    try:
        for tag in results.predictions:
            if tag.tag_name == 'cane toad':
                percentage = tag.probability
    except UnboundLocalError:
        percentage = 'NA'

    return percentage


# setting up project using keys

ENDPOINT = "https://canetoad-prediction.cognitiveservices.azure.com/"

# using hmbur5@student.monash.edu
training_key = "567bc1a3b9d4479283887d68c1d7f46c"
prediction_key = "6b9d10b6c8bc42878d92fe94213256a1"
prediction_resource_id = "/subscriptions/d0bdc746-b59f-4a0c-b651-9865282bcd1a/resourceGroups/CaneToadMachineLearning/providers/Microsoft.CognitiveServices/accounts/CaneToad-Prediction"


credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# finding project id
for project in trainer.get_projects():
    if project.name == 'iterative':
        break
# iteration name must be changed each iteration to publish
publish_iteration_name = "classify_model_basic"

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)


# clear existing training images
trainer.delete_images(project.id, all_images=True, all_iterations=True)
#exit(-1)

# training with image urls
print("Adding images...")



# split labelled data into tuples of training and testing (C for cane toad, N for not)
CtrainingURLS = []
CtestingURLS = []
NtrainingURLS = []
NtestingURLS = []
# getting image urls using ALA file for each species (iterative as we can only upload 64 images at a time)
for species in ['caneToad', 'stripedMarshFrog', 'ornateBurrowingFrog', 'australianGreenTreeFrog', 'bumpyRockFrog',
                'crawlingToadlet', 'daintyGreenTreeFrog', 'desertFroglet', 'desertTreeFrog', 'giantFrog', 'hootingFrog',
                'longFootedFrog', 'marbledFrog', 'moaningFrog', 'motorbikeFrog', 'newHollandFrog', 'rockholeFrog',
                'rothsTreeFrog', 'westernBanjoFrog', 'whiteLippedTreeFrog']:

    if species=='caneToad' or species =='questionCaneToad':
        tag_name = species
    else:
        tag_name = 'otherFrog'


    if species =='caneToad':
        image_url_list = GetALAimages.listOfCheckedImages('ala image urls/confirmedCaneToads.csv')
        CtestingURLS = image_url_list
    elif species=='questionableCaneToad':
        image_url_list = GetALAimages.listOfCheckedImages('ala image urls/confirmedNotCaneToads.csv')
    else:
        file_dir = 'ala image urls/' + species + 'RawFile.csv'
        image_url_list = GetALAimages.listOfAlaImageUrls(file_dir)
        NtestingURLS += image_url_list


# build a control for testing afterwards
CcontrolURLS = []
NcontrolURLS = []
for i in range(0,50)*np.floor(len(CtestingURLS)/50):
    CcontrolURLS.append(CtestingURLS[int(i)])
    CtestingURLS.remove(CtestingURLS[int(i)])
for i in range(0,50)*np.floor(len(NtestingURLS)/50):
    NcontrolURLS.append(NtestingURLS[int(i)])
    NtestingURLS.remove(NtestingURLS[int(i)])



# add 100 of each N and C into training data
CtrainingURLS = np.random.choice(CtestingURLS, 50, replace=False)
NtrainingURLS = np.random.choice(NtestingURLS, 50, replace=False)


addImages(CtrainingURLS, 'cane toad')
addImages(NtrainingURLS, 'other frog')

train(iteration_name='Iteration2.1')


# now refine
iteration = 1

while len(CtestingURLS)>0:
    CtrainingURLS = []
    NtrainingURLS = []
    # test on cane toad images
    for image_url in np.random.choice(CtestingURLS, 100, replace=False):
        # if it doesn't classify as cane toad, add to training data
        prediction = predict(image_url, 'Iteration2.'+str(iteration))
        if prediction == 'NA':
            print(image_url)
            CtestingURLS.remove(image_url)
        elif prediction < 0.95:
            CtrainingURLS.append(image_url)
            CtestingURLS.remove(image_url)

    # test on not cane toad images
    for image_url in np.random.choice(NtestingURLS, 100, replace=False):
        # if it doesn't classify as cane toad, add to training data
        prediction = predict(image_url, 'Iteration2.'+str(iteration))
        if prediction == 'NA':
            print(image_url)
            NtestingURLS.remove(image_url)
        elif prediction > 0.95:
            NtrainingURLS.append(image_url)
            NtestingURLS.remove(image_url)
    addImages(CtrainingURLS, 'cane toad')
    addImages(NtrainingURLS, 'other frog')
    train('Iteration2.'+str(iteration+1))

    iteration+=1




