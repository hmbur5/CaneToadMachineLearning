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
from preCrop import cropImage



def tag(image_url_list, name):
    # create list of cropped regions of frogs or animals from google vision
    image_cropped_list = []
    for image_url in image_url_list:
        crops, tags = cropImage(image_url, checkTags=[name])
        image_cropped_list+=crops
        if len(image_cropped_list)>50:
            return image_cropped_list
    return image_cropped_list


def addImages(image_cropped_list, tag_name):

    # find tag to add images to
    for tag in trainer.get_tags(project.id):
        if tag.name == tag_name:
            break
    if tag.name!= tag_name:
        raise Exception('Tag name does not exist')



    # going through a small portion of url list as can only upload 64 at a time
    for batch_number in range(math.ceil(len(image_cropped_list)/64)):
        print(batch_number)

        image_list = []
        endIndex = (batch_number+1)*64
        if endIndex > len(image_cropped_list):
            endIndex = len(image_cropped_list)
        for cropped_image in image_cropped_list[batch_number*64: endIndex]:
            imgByteArr = BytesIO()
            cropped_image.save(imgByteArr, format='PNG')
            cropped_image = imgByteArr.getvalue()
            image_list.append(ImageFileCreateEntry(contents=cropped_image, tag_ids=[tag.id]))

        # check that there are some images to upload, then upload these
        if len(image_list) > 0:
            upload_result = trainer.create_images_from_files(project.id, ImageUrlCreateBatch(images=image_list))

            # gives error when there is a duplicate (which is possible with the ALA data) so code below is to ignore
            # duplicate error.
            while not upload_result.is_batch_successful:
                image_list = []
                for image in upload_result.images:
                    if image.status == 'OK':
                        # add to new new image_list with corresponding url and tag
                        image_list.append(ImageFileCreateEntry(contents=cropped_image, tag_ids=[tag.id]))
                    elif image.status != 'OKDuplicate':
                            print(image.source_url)
                if len(image_list)>0:
                    upload_result = trainer.create_images_from_files(project.id, ImageUrlCreateBatch(images=image_list))
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


def predict(image_file, iteration_name):
    print(iteration_name)

    # get prediction percentages
    try:
        imgByteArr = BytesIO()
        image_file.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        results = predictor.classify_image(project.id,iteration_name, imgByteArr)

    # not sure what exception should be as custom vision gives a strange error.
    except:
        pass
        '''
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
        '''

    try:
        for tag in results.predictions:
            if tag.tag_name == 'cane toad':
                percentage = tag.probability
    except UnboundLocalError:
        percentage = 'NA'

    return percentage


# setting up project using keys
#nhcha6
ENDPOINT = "https://canetoads-prediction.cognitiveservices.azure.com/"

# using nic
training_key = "741668965a304e89847d9f1f768836f4"
prediction_key = "4e2a8fab822b4d1a93ab372694f99525"
prediction_resource_id = "/subscriptions/5939e776-823a-4dae-bd82-8339288ead8f/resourceGroups/CaneToads/providers/Microsoft.CognitiveServices/accounts/canetoads-Prediction"



credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# finding project id
for project in trainer.get_projects():
    if project.name == 'all':
        break
# iteration name must be changed each iteration to publish
publish_iteration_name = "classify_model_basic"

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)


# clear existing training images
#trainer.delete_images(project.id, all_images=True, all_iterations=True)
#time.sleep(1)
#trainer.delete_images(project.id, all_images=True, all_iterations=True)
#exit(-1)

# training with image urls
print("Adding images...")



# split labelled data into tuples of training and testing (C for cane toad, N for not)
CalaURLS = []
NalaURLS = []
# getting image urls using ALA file for each species (iterative as we can only upload 64 images at a time)
for species in ['caneToad', 'stripedMarshFrog', 'ornateBurrowingFrog', 'australianGreenTreeFrog', 'bumpyRockFrog',
                'crawlingToadlet', 'daintyGreenTreeFrog', 'desertFroglet', 'desertTreeFrog', 'giantFrog', 'hootingFrog',
                'longFootedFrog', 'marbledFrog', 'moaningFrog', 'motorbikeFrog', 'newHollandFrog', 'rockholeFrog',
                'rothsTreeFrog', 'westernBanjoFrog', 'whiteLippedTreeFrog']:


    file_dir = 'ala image urls/' + species + 'RawFile.csv'
    image_url_list = GetALAimages.listOfAlaImageUrls(file_dir)
    if species=='caneToad':
        CalaURLS += image_url_list
    else:
        NalaURLS += image_url_list

# build a control for testing afterwards
CcontrolURLS = []
NcontrolURLS = []
for i in range(0,50)*np.floor(len(CalaURLS)/51):
    CcontrolURLS.append(CalaURLS[int(i)])
    CalaURLS.remove(CalaURLS[int(i)])
for i in range(0,50)*np.floor(len(NalaURLS)/51):
    NcontrolURLS.append(NalaURLS[int(i)])
    NalaURLS.remove(NalaURLS[int(i)])



# start by adding all images that come up as frogs.
#addImages(tag(CalaURLS, 'Frog'), 'cane toad')
#addImages(tag(NalaURLS, 'Frog'), 'other frog')
#exit(-1)


# iteratively build up model of photos that don't get a frog tag from google, checking before adding them to training
# set if they are predicted less than 0.95 cane toad
# we could add all the other frog images here as they don't really need verifying or cropping, but we want to keep the
# number of photos balanced

CtestingURLS = tag(CalaURLS, 'Animal')
NtestingURLS = tag(NalaURLS, 'Animal')


# now refine
iteration = 1

while len(CtestingURLS)>0:

    print(iteration)

    CtrainingURLS = []
    NtrainingURLS = []
    # test on cane toad images, and unless 95% sure these are cane toads, we check the photo before adding to training
    # (as some ala images contain tadpoles and skeletons)
    for image_url in CtestingURLS[0:50]:
        # if it doesn't classify as cane toad, add to training data
        prediction = predict(image_url, 'Iteration1.'+str(iteration))
        print(prediction)
        if prediction == 'NA':
            print(image_url)
            CtestingURLS.remove(image_url)

        elif prediction < 0.95:
            if GetALAimages.manualConfirmationOfTest(image_url):
                # add to training and remove from testing
                CtrainingURLS.append(image_url)
                CtestingURLS.remove(image_url)
            # if not a good cane toad image, simply remove this
            else:
                CtestingURLS.remove(image_url)
        elif prediction>= 0.95:
            # add to training and remove from testing
            CtrainingURLS.append(image_url)
            CtestingURLS.remove(image_url)

    for image_url in NtestingURLS[0:50]:
        # add to training and remove from testing
        NtrainingURLS.append(image_url)
        NtestingURLS.remove(image_url)

    addImages(CtrainingURLS, 'cane toad')
    addImages(NtrainingURLS, 'other frog')

    iteration+=1
    train(iteration_name='Iteration1.' + str(iteration))





