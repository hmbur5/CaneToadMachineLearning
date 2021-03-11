import numpy as np
import matplotlib.pyplot as plt
from preCrop import cropImage
from instaloader import Instaloader
import urllib
import pandas as pd
from GetALAimages import listOfAlaImageUrls
from flickrapi import FlickrAPI
import csv
from predictFromImageUrl import predictFromImageUrl
import time
import datetime
from PIL import Image
import pandas as pd



def createTagsFiles(image_url_list, file_name):
    # first delete duplicate images
    new_image_url_list = []
    open_images = []
    for image_url in image_url_list:
        try:
            img = urllib.request.urlopen(image_url)
            img = Image.open(img)
            if img not in open_images:
                open_images.append(img)
                new_image_url_list.append(image_url)
            else:
                pass
                print('duplicate')
        except urllib.error.HTTPError:
            print(image_url)
        except OSError:
            print(image_url)
        except AttributeError as e:
            print(image_url)
            print(e)
        except UnicodeEncodeError:
            print(image_url)
    image_url_list = new_image_url_list
    if len(image_url_list)>500:
        image_url_list = image_url_list[0:500]


    url_and_tags = []
    print(len(image_url_list))
    for image_url in image_url_list:
        #print(image_url)
        tagsList = []
        try:
            crops, tags = cropImage(image_url, return_coords=True)
            maxArea = 0
            maxCoords = [0,0,0,0]
            for image, coords in crops:
                x1,y1,x2,y2,ratio = coords
                if abs((x1-x2)*(y1-y2))>maxArea:
                    maxCoords = [x1,y1,x2,y2]
                    maxArea = abs((x1-x2)*(y1-y2))
            tagsList += tags



        except urllib.error.HTTPError:
            pass
        except urllib.error.URLError:
            pass
        except TypeError:
            pass
        #print(tagsList)
        url_and_tags.append([image_url, tagsList, maxCoords])

    # write new file with image urls and prediction percentages
    with open('tags/' + file_name + '.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(url_and_tags)

def createTagsFilesFromImages(folder, file_name):
    from tqdm import tqdm
    import os
    from google.cloud.vision import types
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    image_file_list = []
    for i in tqdm(os.listdir(folder)):
        path = os.path.join(folder, i)
        with open(path, 'rb') as image_file:
            content = image_file.read()
        image = types.Image(content=content)
        image_file_list.append([image, path])

    print(len(image_file_list))

    url_and_tags = []

    for image, image_dir in image_file_list:
        # Performs label detection on the image file
        response = client.object_localization(image=image)
        tagsList = []
        maxArea = 0
        maxCoords = [0, 0, 0, 0]
        for tag in response.localized_object_annotations:
            tagsList.append(tag.name)
            vertices = tag.bounding_poly.normalized_vertices
            coords = [vertices[0].x, vertices[0].y,
                      vertices[2].x, vertices[2].y]
            x1, y1, x2, y2 = coords
            if abs((x1-x2)*(y1-y2)) > maxArea:
                maxCoords = coords
                maxArea = abs((x1 - x2) * (y1 - y2))
        url_and_tags.append([image_dir, tagsList, maxCoords])

    # write new file with image urls and prediction percentages
    with open('tags/' + file_name + '.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(url_and_tags)



def getTagsFromFile(file_name):
    url_and_tags = []
    with open('tags/' + file_name +'.csv', "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
            url = lines[0]
            tags = lines[2]
            tags = tags[1:-1] # remove [ and ] from string
            if len(tags)==0:
                newTags = []
            else:
                tags = list(tags.split(", ")) # convert back to list
                # remove quotation marks from each string
                newTags = []
                for tag in tags:
                    newTags.append(tag[1:-1])
            coords = lines[1]
            coords = coords[1:-1]  # remove [ and ] from string
            coords = list(coords.split(", "))  # convert back to list
            for index, coord in enumerate(coords):
                coords[index] = float(coord)
            # remove quotation marks from each string
            url_and_tags.append([url, coords, newTags])
    return url_and_tags


def getTagsFromPredictions(file_name, file_dir, return_note=False):
    url_and_tags = []
    with open('predictions/'+file_dir+'/' + file_name +'_tag_predictions.csv', "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
            # skip over first line
            if lines[0]=='url' or lines[0]=='':
                continue

            url = lines[0]
            if pd.read_csv('predictions/'+file_dir+'/' + file_name +'_tag_predictions.csv', nrows=1).columns[1]=='best crop':
                tags=lines[2]
            else:
                tags = lines[1]
            if tags[0]!= '[':
                continue
            tags = tags[1:-1] # remove [ and ] from string
            if len(tags)==0:
                newTags = []
            else:
                tags = list(tags.split(", ")) # convert back to list
                # remove quotation marks from each string
                newTags = []
                for tag in tags:
                    newTags.append(tag[1:-1])
            # getting best crop
            coords = 'NA'# lines[1]
            if coords!='NA':
                coords = coords[1:-1]  # remove [ and ] from string
                coords = list(coords.split(", "))  # convert back to list
                for index, coord in enumerate(coords):
                    coords[index] = float(coord)
            prediction = 0#float(lines[4])
            try:
                note=lines[6]
            except IndexError:
                index=4
            else:
                index=5

            reid = lines[index]
            if not return_note:
                url_and_tags.append([url, newTags, coords, prediction, reid])
            else:
                try:
                    note = lines[index+1]
                except IndexError:
                    note = ''
                url_and_tags.append([url, newTags, coords, prediction, reid, note])

    return url_and_tags



def createPredictionFiles(file_name):
    testing_image_urls = []
    with open('tags/' + file_name +'.csv', "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
            url = lines[0]
            tags = lines[1]
            # avoid any duplicates
            if [url, tags, 'NA', 'NA', 'NA'] not in testing_image_urls:
                testing_image_urls.append([url, tags, 'NA', 'NA', 'NA'])
            # just use first 250 images
            if len(testing_image_urls)>=400:
                break
    save_file_as =  file_name + '_tag_predictions'
    predictFromImageUrl(testing_image_urls, save_file_as)



if __name__ == '__main__':

    #for file_name in ['instgramCaneToad_all']:
    #    createPredictionFiles(file_name)

    #exit(-1)

    # facebook
    imageUrls = []
    with open('facebook_cane_toad_search/facebook_cane_toad_search.html') as file:
        for line in file:
            imageUrls.append(line[10:-4])
    # facebook doesn't work with image urls
    #createTagsFiles(imageUrls, 'facebook')
    #exit(-1)

    # twitter
    imageUrls = []
    with open('tags/twitter_german_wasp.html') as file:
        for line in file:
            imageUrls.append(line[10:-4])

    #createTagsFiles(imageUrls, 'twitter_german_wasp')

    #exit(-1)


    # flickr
    # setting up flickr api
    FLICKR_PUBLIC = '67b52264b7ded98bd1af796bb92b5a14'
    FLICKR_SECRET = '5c7c9c7344542522'

    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
    # extras for photo search (can include geo tag, date etc)
    extras = 'geo, url_t, url_c, date_upload'

    photoUrls = []
    for pageNumber in [0,1]:
        # search limited to those with gps coordinates within australia
        photoSearch = flickr.photos.search(text='german wasp', per_page=250, page=pageNumber, has_geo = True, extras=extras)#,
                                           #bbox='113.338953078, -43.6345972634, 153.569469029, -10.6681857235')
        photos = photoSearch['photos']
        for element in photos['photo']:
            try:
                photoUrls.append(element['url_c'])
            except:
                # if larger image file doesn't exist, just use thumbnail
                photoUrls.append(element['url_t'])
    #createTagsFiles(photoUrls, 'flickr_german_wasp')
    #exit(-1)

    # getting random photos from flickr
    import random
    photoUrls = []
    '''
    for i in range(300):
        start_date = datetime.date(2020, 1, 1)
        end_date = datetime.date(2021, 1, 1)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)
        random_date = datetime.datetime.combine(random_date, datetime.time(0, 0))
        random_date = datetime.datetime.timestamp(random_date)

        photoSearch = flickr.photos.search(max_upload_date=random_date, sort='date-posted-desc', extras=extras)
        photos = photoSearch['photos']
        for element in photos['photo']:
            try:
                photoUrls.append(element['url_c'])
                print(element['url_c'])
            except Exception as e:
                print(e)
                continue
            else:
                break
    createTagsFiles(photoUrls, 'random')
'''
    #ala
    images=listOfAlaImageUrls('ala image urls/caneToadRawFile.csv')
    createTagsFiles(images[0:600],'ala_cane_toad')
    exit(-1)


    # inaturalist
    df = pd.read_csv("ala image urls/iNaturalist_european_wasp.csv")
    saved_column = list(df['image_url'])
    #createTagsFiles(saved_column[0:250], 'inaturalist_european_wasp')
    #exit(-1)


    # instagram
    loader = Instaloader()
    NUM_POSTS = 10


    def get_hashtags_posts(mainTag, maxCount, additionalTag=None):
        posts = loader.get_hashtag_posts(mainTag)
        urls = []
        count = 0
        for post in posts:
            # skip all posts from december/november
            #if post.date>datetime.datetime(2020, 10, 28):
            #    continue
            print(post.date)
            time.sleep(1)
            if not additionalTag or additionalTag in post.caption_hashtags:
                urls.append(post.url)
                count += 1
                if count == maxCount:
                    return urls

    #justCaneToad = get_hashtags_posts('germanwasp', 400)
    #caneToadAndFrog = get_hashtags_posts('canetoad', 500, 'frog')
    #caneToadAndAmphibian = get_hashtags_posts('amphibian', 500, 'frog')

    #createTagsFiles(justCaneToad, 'instagram_german_wasp')
    #createTagsFiles(caneToadAndFrog, 'instgramCaneToadAndFrog')
    #createTagsFiles(caneToadAndAmphibian, 'instgramCaneToadAndAmphibian')
    #exit(-1)



    # reddit
    import praw

    reddit = praw.Reddit(client_id='taeY_V0qktbKRg', client_secret='FCXYgqAcZ3vjTOrID52UOPiDqBk', user_agent='canetoad')
    all = reddit.subreddit('all')
    reddit_url_list = []
    for b in all.search("german wasps", limit=500):
        try:
            # if image urls are in metadata
            for key in b.media_metadata.keys():
                # add image url for each image
                reddit_url_list.append(b.media_metadata[key]['s']['u'])
        except AttributeError:
            # else if image url is in the submission
            reddit_url_list.append(b.url)

    createTagsFiles(reddit_url_list, 'reddit_german_wasps')





