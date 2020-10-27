import numpy as np
import matplotlib.pyplot as plt
from preCrop import cropImage
from instaloader import Instaloader
import urllib
import pandas as pd
from GetALAimages import listOfAlaImageUrls
from flickrapi import FlickrAPI
import csv


def createTagsFiles(image_url_list, file_name):
    url_and_tags = []
    print(len(image_url_list))
    for image_url in image_url_list:
        tagsList = []
        try:
            crops, tags = cropImage(image_url)
            tagsList += list(set(tags))
        except urllib.error.HTTPError:
            pass
        except TypeError:
            pass
        url_and_tags.append([image_url, tagsList])

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
        for tag in response.localized_object_annotations:
            tagsList.append(tag.name)
        url_and_tags.append([image_dir, tagsList])

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
            tags = lines[1]
            tags = tags[1:-1] # remove [ and ] from string
            if len(tags)==0:
                url_and_tags.append([url, []])
            else:
                tags = list(tags.split(", ")) # convert back to list
                # remove quotation marks from each string
                newTags = []
                for tag in tags:
                    newTags.append(tag[1:-1])
                url_and_tags.append([url, newTags])
    return url_and_tags

if __name__ == '__main__':

    # facebook
    createTagsFilesFromImages('facebook_cane_toad_search', 'facebook')

    # twitter
    createTagsFilesFromImages('twitter_canetoad_hashtag', 'twitter')

    exit(-1)


    # flickr
    # setting up flickr api
    FLICKR_PUBLIC = '67b52264b7ded98bd1af796bb92b5a14'
    FLICKR_SECRET = '5c7c9c7344542522'

    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
    # extras for photo search (can include geo tag, date etc)
    extras = 'geo, url_t, url_c, date_taken'

    photoUrls = []
    for pageNumber in [0,1]:
        # search limited to those with gps coordinates within australia
        photoSearch = flickr.photos.search(text='cane toad', per_page=250, page=pageNumber, has_geo = True, extras=extras,
                                           bbox='113.338953078, -43.6345972634, 153.569469029, -10.6681857235')
        photos = photoSearch['photos']
        for element in photos['photo']:
            try:
                photoUrls.append(element['url_c'])
            except:
                # if larger image file doesn't exist, just use thumbnail
                photoUrls.append(element['url_t'])
    createTagsFiles(photoUrls, 'flickr')
    #exit(-1)

    #ala
    images=listOfAlaImageUrls('ala image urls/caneToadRawFile.csv')
    createTagsFiles(images[0:500],'ala')


    # inaturalist
    df = pd.read_csv("ala image urls/iNaturalist cane toad.csv")
    saved_column = list(df['image_url'])
    createTagsFiles(saved_column[0:500], 'inaturalist')


    # instagram
    loader = Instaloader()
    NUM_POSTS = 10

    def get_hashtags_posts(query):
        posts = loader.get_hashtag_posts(query)
        urls = []
        count = 0
        for post in posts:
            urls.append(post.url)
            count+=1
            if count==500:
                return urls

    createTagsFiles(get_hashtags_posts('canetoad'), 'instgram')
    #exit(-1)


    # reddit
    import praw
    reddit = praw.Reddit(client_id='taeY_V0qktbKRg', client_secret='FCXYgqAcZ3vjTOrID52UOPiDqBk', user_agent='canetoad')
    all = reddit.subreddit('all')
    reddit_url_list = []
    for b in all.search("cane toads", limit=500):
        reddit_url_list.append(b.url)

    createTagsFiles(reddit_url_list, 'reddit')




