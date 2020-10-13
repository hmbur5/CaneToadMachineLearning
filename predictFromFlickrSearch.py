from flickrapi import FlickrAPI
from predictFromImageUrl import predictFromImageUrl


# setting up flickr api
FLICKR_PUBLIC = '67b52264b7ded98bd1af796bb92b5a14'
FLICKR_SECRET = '5c7c9c7344542522'

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
# extras for photo search (can include geo tag, date etc)
extras='geo, url_t, url_c, date_taken'


# iterating through different search queries and saving the predictions to a file.
searchQueries = ['cane toad', 'frog', 'toad', 'magpie', 'lizard']
for search in searchQueries:
    print(search)
    photoUrls = []
    # search limited to those with gps coordinates within australia
    photoSearch = flickr.photos.search(text=search, per_page=250, has_geo = True, extras=extras,
                                       bbox='113.338953078, -43.6345972634, 153.569469029, -10.6681857235')
    photos = photoSearch['photos']
    for element in photos['photo']:
        try:
            photoUrls.append([element['url_c'], 'flickr_'+ search, element['latitude'], element['longitude'], element['datetaken']])
        except:
            # if larger image file doesn't exist, just use thumbnail
            photoUrls.append([element['url_t'], 'flickr_' + search, element['latitude'], element['longitude'], element['datetaken']])

    predictFromImageUrl(photoUrls, 'flickr_'+search)



    exit(-1)
    # creating histogram of tags from flickr
    import matplotlib.pyplot as plt
    from preCrop import cropImage
    import numpy as np
    tagsList = []
    for photo in photoUrls:
        image_url = photo[0]
        crops, tags = cropImage(image_url)
        tagsList+=tags
    labels, counts = np.unique(tagsList, return_counts=True)
    sorted_indices = np.argsort(-counts)
    ticks = range(len(counts[sorted_indices]))
    plt.bar(ticks, counts[sorted_indices], align='center')
    plt.xticks(ticks, labels[sorted_indices], rotation='vertical')
    plt.show()

