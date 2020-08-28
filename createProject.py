# custom vision documentation can be found here:
# https://azuresdkdocs.blob.core.windows.net/$web/python/azure-cognitiveservices-vision-customvision/3.0.0/azure.cognitiveservices.vision.customvision.prediction.html#module-azure.cognitiveservices.vision.customvision.prediction


from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials

ENDPOINT = "https://canetoadmachinelearning.cognitiveservices.azure.com/"

# Replace with a valid key
training_key = "cde7deba2d5d4df5b768b50b700c46b7"
prediction_key = "fb49a542a16a47e6b68b2983db158c32"
prediction_resource_id = "/subscriptions/baa59b08-5ec4-44ea-a907-b12782d8e2a0/resourceGroups/Canetoads/providers/Microsoft.CognitiveServices/accounts/CaneToadMachineLea-Prediction"

publish_iteration_name = "classifyModel"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)


# Create a new project
print ("Creating project...")
project = trainer.create_project("Cane Toad Classifier Binary")


# Make two tags in the new project
canetoad_tag = trainer.create_tag(project.id, "cane toad")
negative_tag = trainer.create_tag(project.id, "not cane toad", type='Negative')


base_image_url = "alaPhotos/"

print("Adding images...")

image_list = []

for image_num in range(0, 5):
    file_name = "canetoad_{}.jpg".format(image_num)
    with open(base_image_url + "canetoad/" + file_name, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[canetoad_tag.id]))

for image_num in range(0, 5):
    file_name = "frog_{}.jpg".format(image_num)
    with open(base_image_url + "frog/" + file_name, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[negative_tag.id]))

upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
if not upload_result.is_batch_successful:
    print("Image batch upload failed.")
    for image in upload_result.images:
        print("Image status: ", image.status)
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



from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

with open(base_image_url + "test_image.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        project.id, publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))