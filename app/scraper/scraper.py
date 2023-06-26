import requests
from bs4 import BeautifulSoup


def getDataFromWebsite(url):
    r = requests.get(url)
    return r.text

# Funkcja prepareData(tag) pobiera dane ze wskazanej strony i zwraca,
# jako tablicę zawierające URL zdjęć
def prepareData(tag):
    target_url = "https://unsplash.com/s/photos/" + tag
    all_data = getDataFromWebsite(target_url)

    soup = BeautifulSoup(all_data, 'html.parser')

    # Find links with correct photos
    photo_links = []
    for item in soup.find_all('img'):
        src = item.get('src')
        if "/photo/" in src:
            photo_links.append(src)

    return photo_links
