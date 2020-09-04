from flickrapi import FlickrAPI
from predictFromImageUrl import predictFromImageUrl


# setting up flickr api
FLICKR_PUBLIC = '67b52264b7ded98bd1af796bb92b5a14'
FLICKR_SECRET = '5c7c9c7344542522'

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
# extras for photo search (can include geo tag, date etc)
extras='url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'


# iterating through different search queries and saving the predictions to a file.
searchQueries = ['frog', 'toad', 'cane toad', 'magpie']
for search in searchQueries:
    print(search)
    photoUrls = []
    photoSearch = flickr.photos.search(text=search, per_page=50, extras=extras)
    photos = photoSearch['photos']
    for element in photos['photo']:
        try:
            photoUrls.append([element['url_c'], 'flickr_'+search])
        except:
            # if larger image file doesn't exist, just use thumbnail
            photoUrls.append([element['url_t'], 'flickr_' + search])

    predictFromImageUrl(photoUrls, 'flickr_'+search)
