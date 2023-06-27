import numpy as np
import tensorflow as tf
import os
import cv2
import random
import urllib.request
from app.scraper.scraper import prepareData
import keras.utils as image

def getDataFromScraper():
    rose = prepareData("rose")
    daisy = prepareData("daisy")
    dandelion = prepareData("dandelion")
    sunflower = prepareData("sunflower")
    tulip = prepareData("tulip")

    random_rose = random.choice(rose)
    random_daisy = random.choice(daisy)
    random_dandelion = random.choice(dandelion)
    random_sunflower = random.choice(sunflower)
    random_tulip = random.choice(tulip)

    return random_rose, random_daisy, random_dandelion, random_sunflower, random_tulip


def main():
    # Pobierz dane ze scrapera
    img_urls = getDataFromScraper()

    # Ustawienie kategorii danych
    flower_categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    # Wytrenowany model
    model = tf.keras.models.load_model('C:/Users/Kamil/Desktop/Projekt_SI/app/assets/model/flowers.h5')

    # Zmienne pomocnicze
    num = 0
    scraper_images_path = 'C:/Users/Kamil/Desktop/Projekt_SI/app/assets/scraper_images/'
    random_images = []

    # Pokaż zdj pobrane ze scrapera
    for img_url in img_urls:
        num = num + 1
        img_name = f'image_{num}.jpg'
        img_path = os.path.join(scraper_images_path, img_name)
        urllib.request.urlretrieve(img_url, img_path)
        img = cv2.imread(img_path)
        cv2.imshow('img', img)
        key = cv2.waitKey(0)
        random_images.append(img_path)

        # Enter spowoduje wybranie orazu poddanego dalszej analizie
        if key == 13:
            break


    for i in range(len(random_images)):
        # Załadowanie obrazu i nadanie mu odpowiedniego rozmiaru
        test_image = image.load_img(random_images[i], target_size=(224, 224))

        # Konwertowanie wczytanego obrazu na tablicę wielowymiarową
        test_image = image.img_to_array(test_image)

        # Tworzenie trzywymiarowej tablicy
        test_image = np.expand_dims(test_image, axis=0)

        # Przewidywanie kategorii obrazu (wykorzystując wytrenowany model)
        result = model.predict(test_image)

        # Znalezienie pozycji elementu o najwyżśzej wartości predykcji
        indPositionMax = np.argmax(result[0])

        # Przypisanie przewidzianej kategorii do zmiennej
        flower_predict = flower_categories[indPositionMax]

        # Wczytanie obrazu
        imgFinalResult = cv2.imread(random_images[i])
        font = cv2.FONT_HERSHEY_COMPLEX

        # Przygotowanie wyświetlenia wyniku
        cv2.putText(imgFinalResult, flower_predict, (7, 25), font, 1, (191, 255, 0), 2)
        cv2.imshow('img', imgFinalResult)
        cv2.waitKey(0)

        # Zapisanie wyniku
        result_path = os.path.join(scraper_images_path+f'result_{i}.jpg')
        cv2.imwrite(result_path, imgFinalResult)



if __name__ == '__main__':
    main()
