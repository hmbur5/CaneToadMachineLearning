from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateBatch, ImageUrlCreateEntry
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import GetALAimages
import math


# setting up project using keys

ENDPOINT = "https://canetoadmachinelearning.cognitiveservices.azure.com/"

training_key = "cde7deba2d5d4df5b768b50b700c46b7"
prediction_key = "fb49a542a16a47e6b68b2983db158c32"
prediction_resource_id = "/subscriptions/baa59b08-5ec4-44ea-a907-b12782d8e2a0/resourceGroups/Canetoads/providers/Microsoft.CognitiveServices/accounts/CaneToadMachineLea-Prediction"


credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# finding project id
for project in trainer.get_projects():
    if project.name == 'Cane Toad Classifier Python':
        break
# iteration name must be changed each iteration to publish
publish_iteration_name = "classifyModel_large"




# training with image urls
print("Adding images...")


# getting image urls using ALA file for each species (iterative as we can only upload 64 images at a time)
for species in ['caneToad', 'stripedMarshFrogs']:

    # finding tag id
    for tag in trainer.get_tags(project.id):
        if tag.name == species:
            break

    file_dir = 'ala image urls/' + species + 'RawFile.csv'
    image_url_list = GetALAimages.listOfAlaImageUrls(file_dir)
    for batch_number in range(math.floor(len(image_url_list)/64)):
        image_list = []
        # going through a small portion of list as can only do 64 at a time
        for url in image_url_list[batch_number*64:(batch_number+1)*64]:
            image_list.append(ImageUrlCreateEntry(url=url, tag_ids=[tag.id]))


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
                    #exit(-1)
            if len(image_list)>0:
                upload_result = trainer.create_images_from_urls(project.id, ImageUrlCreateBatch(images=image_list))
            else:
                break




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

