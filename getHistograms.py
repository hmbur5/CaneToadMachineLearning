import numpy as np
import matplotlib.pyplot as plt
from preCrop import cropImage
from instaloader import Instaloader
import urllib

def histogram(image_url_list):
    print(len(image_url_list))
    tagsList = []
    for image_url in image_url_list:
        try:
            crops, tags = cropImage(image_url)
            tagsList += list(set(tags))
        except urllib.error.HTTPError:
            pass
    labels, counts = np.unique(tagsList, return_counts=True)
    sorted_indices = np.argsort(-counts)
    ticks = range(len(counts[sorted_indices]))
    counts = counts/max(counts)
    plt.bar(ticks, counts[sorted_indices], align='center')
    plt.xticks(ticks, labels[sorted_indices], rotation='vertical')
    plt.show()




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
        if count==200:
            return urls

histogram(get_hashtags_posts('canetoad'))
#exit(-1)


# reddit
import praw
reddit = praw.Reddit(client_id='XflYI0t3JONLqw', client_secret='385A2Q2YjA9N7HPo0PFfl8raD0U', user_agent='canetoad')
all = reddit.subreddit('all')
reddit_url_list = []
for b in all.search("cane toads", limit=200):
    reddit_url_list.append(b.url)
histogram(reddit_url_list)

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
histogram(twitter_urls)


