from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateBatch, ImageUrlCreateEntry
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import GetALAimages
import math
import csv


# setting up project using keys

ENDPOINT = "https://canetoadmachinelearning.cognitiveservices.azure.com/"

training_key = "cde7deba2d5d4df5b768b50b700c46b7"
prediction_key = "fb49a542a16a47e6b68b2983db158c32"
prediction_resource_id = "/subscriptions/baa59b08-5ec4-44ea-a907-b12782d8e2a0/resourceGroups/Canetoads/providers/Microsoft.CognitiveServices/accounts/CaneToadMachineLea-Prediction"


credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# finding project id
for project in trainer.get_projects():
    if project.name == 'Cane Toad Classifier Binary':
        break
# iteration name must be changed each iteration to publish
publish_iteration_name = "classify_model_witholding_some_images"



# clear existing training images
#trainer.delete_images(project.id, all_images=True, all_iterations=True)

# training with image urls
print("Adding images...")

# initialise list which stores a bunch of labelled images for testing later
test_list = []

# getting image urls using ALA file for each species (iterative as we can only upload 64 images at a time)
for species in ['caneToad', 'stripedMarshFrog', 'ornateBurrowingFrog', 'australianGreenTreeFrog', 'bumpyRockFrog',
                'crawlingToadlet', 'daintyGreenTreeFrog', 'desertFroglet', 'desertTreeFrog', 'giantFrog', 'hootingFrog',
                'longFootedFrog', 'marbledFrog', 'moaningFrog', 'motorbikeFrog', 'newHollandFrog', 'rockholeFrog',
                'rothsTreeFrog', 'westernBanjoFrog', 'whiteLippedTreeFrog']:

    # finding tag id
    for tag in trainer.get_tags(project.id):
        if species=='caneToad':
            if tag.name == 'cane toad':
                break
        else:
            # replacing species tag with Negative for non-cane toads
            if tag.name== 'not cane toad':
                break


    file_dir = 'ala image urls/' + species + 'RawFile.csv'
    image_url_list = GetALAimages.listOfAlaImageUrls(file_dir)

    # going through a small portion of url list as can only upload 64 at a time
    for batch_number in range(math.ceil(len(image_url_list)/67)):
        image_list = []
        endIndex = (batch_number+1)*67
        if endIndex > len(image_url_list):
            endIndex = len(image_url_list)
        for url in image_url_list[batch_number*67: endIndex-3]:
            image_list.append(ImageUrlCreateEntry(url=url, tag_ids=[tag.id]))
        # every 67 items, add 3 to the testing images, along with their actual species
        for i in range(endIndex-3,endIndex):
            if i>=0 and i<len(image_url_list):
                test_list.append([image_url_list[i], species])

        # check that there are some images to train on, then upload these
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
                        print("Image batch upload failed.")
                        print("Image status: ", image.status)
                        print("Image url: ", image.source_url)
                if len(image_list)>0:
                    upload_result = trainer.create_images_from_urls(project.id, ImageUrlCreateBatch(images=image_list))
                else:
                    break



# save urls for testing to a csv file
#test_list = [[el] for el in test_list]
with open('predictions/binaryAll.csv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter = ',')
    wr.writerows(test_list)

exit(-1)

import time

print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Publish it to the project endpoint

trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")

