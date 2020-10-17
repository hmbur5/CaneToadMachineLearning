from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateBatch, ImageUrlCreateEntry
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import GetALAimages
import math
import csv


# setting up project using keys


# setting up project using keys
ENDPOINT = "https://canetoadmodel-prediction.cognitiveservices.azure.com/"

# using nic
training_key = "af562773faaa490eb5028a14ded3b8cc"
prediction_key = "8043e5cca5634caaab92697bf568942d"
prediction_resource_id = "/subscriptions/79ac0136-fad4-4fe7-bda8-aec4a67de458/resourceGroups/CaneToads/providers/Microsoft.CognitiveServices/accounts/CaneToadModel-Prediction"


credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# finding project id
for project in trainer.get_projects():
    if project.name == 'detection':
        break
# iteration name must be changed each iteration to publish
publish_iteration_name = "classify_model_basic"



# clear existing training images
trainer.delete_images(project.id, all_images=True, all_iterations=True)
#exit(-1)

# training with image urls
print("Adding images...")

# initialise list which stores a bunch of labelled images for testing later
test_list = []

# getting image urls using ALA file for each species (iterative as we can only upload 64 images at a time)
for species in ['caneToad', 'stripedMarshFrog', 'ornateBurrowingFrog', 'australianGreenTreeFrog', 'bumpyRockFrog',
                'crawlingToadlet', 'daintyGreenTreeFrog', 'desertFroglet', 'desertTreeFrog', 'giantFrog', 'hootingFrog',
                'longFootedFrog', 'marbledFrog', 'moaningFrog', 'motorbikeFrog', 'newHollandFrog', 'rockholeFrog',
                'rothsTreeFrog', 'westernBanjoFrog', 'whiteLippedTreeFrog']:

    for tag in trainer.get_tags(project.id):
        if tag.name == species:
            break
    # finding tag id
    if tag.name != species:
        tag = trainer.create_tag(project.id, species)



    if species =='caneToad':
        image_url_list = GetALAimages.listOfCheckedImages('ala image urls/confirmedCaneToads.csv')
    elif species=='questionableCaneToad':
        image_url_list = GetALAimages.listOfCheckedImages('ala image urls/confirmedNotCaneToads.csv')
    else:
        file_dir = 'ala image urls/' + species + 'RawFile.csv'
        image_url_list = GetALAimages.listOfAlaImageUrls(file_dir)


    # going through a small portion of url list as can only upload 64 at a time
    for batch_number in range(math.ceil(len(image_url_list)/64)):

        # removing every third batch to reduce number of not cane toads, to get balanced data
        #if species!='caneToad' and species!='questionableCaneToad':
        #    if batch_number%3==1:
        #        break

        image_list = []
        endIndex = (batch_number+1)*64
        if endIndex > len(image_url_list):
            endIndex = len(image_url_list)
        for url in image_url_list[batch_number*64: endIndex]:
            image_list.append(ImageUrlCreateEntry(url=url, tag_ids=[tag.id]))
        # every 67 items, add 3 to the testing images, along with their actual species
        #for i in range(endIndex-3,endIndex):
        #    if i>=0 and i<len(image_url_list):
        #        test_list.append([image_url_list[i], species])

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
#with open('predictions/binaryAll.csv', 'w') as myfile:
#    wr = csv.writer(myfile, delimiter = ',')
#    wr.writerows(test_list)

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

