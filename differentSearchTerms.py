def get_hashtags_posts(mainTag,  maxCount, additionalTag=None):
    posts = loader.get_hashtag_posts(mainTag)
    urls = []
    count = 0
    for post in posts:
        if not additionalTag or additionalTag in post.caption_hashtags:
            urls.append(post.url)
            count += 1
            if count == maxCount:
                return urls



from instaloader import Instaloader
loader = Instaloader()

l1 = get_hashtags_posts('canetoad',500, 'amphibian')
print(l1)