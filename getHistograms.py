import numpy as np
import matplotlib.pyplot as plt
from preCrop import cropImage
from instaloader import Instaloader
import urllib
import pandas as pd
from GetALAimages import listOfAlaImageUrls
from flickrapi import FlickrAPI


def histogram(image_url_list, source):
    print(len(image_url_list))
    tagsList = []
    for image_url in image_url_list:
        try:
            crops, tags = cropImage(image_url)
            tagsList += list(set(tags))
        except urllib.error.HTTPError:
            pass
        except TypeError:
            pass
    labels, counts = np.unique(tagsList, return_counts=True)
    # crop to first 15 tags
    sorted_indices = np.argsort(-counts)
    counts = counts[sorted_indices]
    labels = labels[sorted_indices]
    counts = counts[0:25]
    labels = labels[0:25]
    ticks = range(len(counts))
    # normalise to proportion of total number of images
    counts = counts/len(image_url_list)
    plt.bar(ticks, counts, align='center')
    plt.xticks(ticks, labels, rotation='vertical')
    plt.title(source)
    plt.show()

if __name__ == '__main__':


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
    histogram(photoUrls, 'flickr')
    #exit(-1)

    #ala
    images=listOfAlaImageUrls('ala image urls/caneToadRawFile.csv')
    histogram(images[0:500],'ala')


    # inaturalist
    df = pd.read_csv("ala image urls/iNaturalist cane toad.csv")
    saved_column = list(df['image_url'])
    histogram(saved_column[0:500], 'inaturalist')


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

    histogram(get_hashtags_posts('canetoad'), 'instgram')
    #exit(-1)


    # reddit
    import praw
    reddit = praw.Reddit(client_id='taeY_V0qktbKRg', client_secret='FCXYgqAcZ3vjTOrID52UOPiDqBk', user_agent='canetoad')
    all = reddit.subreddit('all')
    reddit_url_list = []
    for b in all.search("cane toads", limit=500):
        reddit_url_list.append(b.url)
    histogram(reddit_url_list, 'reddit')

    # twitter
    credentials = {
        "consumer_key": "jH7ccXxYEd3kPW6yM6aHFsYBC",
        "consumer_secret": "bE88zLwdSYl0ba2bGUYmYiqJeH96MMCXdt7BBauY7k6tE5I0uW",
        "access_token": "1299940936357535744-os0uaswGW2BBrioMzV2GAWUgkoUqN2",
        "access_token_secret": "iQyzxHn79mCb16tVDt3XzRDHpvfjVwPl7weD4trruEgT4"
    }

    #exit(-1)

    import twitter

    api = twitter.Api(
        consumer_key = credentials["consumer_key"],
        consumer_secret = credentials["consumer_secret"],
        access_token_key = credentials["access_token"],
        access_token_secret = credentials["access_token_secret"])


    # build tweet URL using tweet ID and screen name of tweet's author
    def get_tweet_url(tweet):
        tweet_id = tweet.id_str
        screen_name = tweet.user.screen_name

        tweet_url = "https://twitter.com/{screen_name}/status/{tweet_id}"
        tweet_url = tweet_url.format(screen_name=screen_name, tweet_id=tweet_id)

        return tweet_url


    query='q=cane%20toad&count=100'


    total_iterations = 5

    all_results = []
    max_id = None
    IDs = []

    for i in range(0, total_iterations):

        results = api.GetSearch(raw_query=query)
        print(results)
        all_results.extend(results)
        IDs = [result.id for result in results]
        smallest_ID = min(IDs)

        if max_id == None:  # first call
            max_id = smallest_ID
            query += '&max_id={max_id}'.format(max_id=max_id)
        else:
            old_max_id = "max_id={max_id}".format(max_id=max_id)
            max_id = smallest_ID
            print(max_id)
            new_max_id = "max_id={max_id}".format(max_id=max_id)
            query = query.replace(old_max_id, new_max_id)

    # keep track of image origin info
    image_origins = {
        "tweet_url": [],
        "image_id": [],
        "image_url": [],
    }


    # keep track of IDs of downloaded images to avoid re-downloads
    twitter_urls = []

    for tweet in all_results:

        tweet_url = get_tweet_url(tweet)

        if tweet.media:

            for media in tweet.media:  # a tweet can have multiple images/videos

                twitter_urls.append(media.media_url)


    print(len(twitter_urls))
    twitter_urls=list(set(twitter_urls))
    print(len(twitter_urls))
    histogram(twitter_urls, 'twitter')


