import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Zbiór danych podzielone na trzy podzbiory
TRAIN_DIR = "C:/Users/Kamil/Desktop/Projekt_SI/app/assets/flower_photos/train"
TEST_DIR = "C:/Users/Kamil/Desktop/Projekt_SI/app/assets/flower_photos/test"
VALIDATION_DIR = "C:/Users/Kamil/Desktop/Projekt_SI/app/assets/flower_photos/validation"

# Dane trenujące
# Tworzenie generatora danych trenujących
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# Generowanie danych trenujących
train_set = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=32 , class_mode='categorical')

# Dane walidacyjne
# Tworzenia generatora danych walidacyjnych
val_datagen = ImageDataGenerator(rescale=1. / 255)
# Generowanie danych walidacyujnych
val_set = val_datagen.flow_from_directory(VALIDATION_DIR, target_size=(224,224), batch_size=32 , class_mode='categorical')

# Tworzenie sekwencyjnego modelu Keras składajacego się z 11 wartsw
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=[224,224,3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2) ))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='Same', activation='relu'  ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2) ))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(5,5), padding='Same', activation='relu'  ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2) ))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(5,5), padding='Same', activation='relu'  ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2) ))
model.add(tf.keras.layers.Dropout(0.5))

# Przygotowanie danych przedd warstwą Dense
model.add(tf.keras.layers.Flatten())

# Dodanie warstwy Dense o 512 jednostkach (neuronach) i funkcji aktywacji 'relu'
model.add(tf.keras.layers.Dense (units=512 , activation='relu'))

# Dodanie ostatniej warstwy Dense o 5 jednostkach (neuronach) i funkcji aktywacji 'softmax'
model.add(tf.keras.layers.Dense(units=5 , activation='softmax'))

# Podsumowanie architektury modelu
print(model.summary())

# Kompilacja modelu z optymalizatorem 'rmsprop', funkcją straty 'categorical_crossentropy' i metryką 'accuracy'
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu na danych trenujących i ocena na danych walidacyjnych przez 20 epok
history = model.fit (x=train_set, validation_data=val_set, batch_size=32 , epochs=20)

# Dane z treningu
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
print(acc)
print(val_acc)

# Wyświetlenie wyników przy użyciu pyplot
epochs_range = range(20)
plt.figure(figsize=(8,8))

# Wykres dokładności treningu i walidacji
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = "Training Accuracy")
plt.plot(epochs_range, val_acc, label = "Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and validation Accuracy')

# Wykres straty treningu i walidacji
plt.subplot(1,2,2)
plt.plot(epochs_range, loss ,label = "Training Loss")
plt.plot(epochs_range, val_loss, label = "Validation Loss")
plt.legend(loc='upper right')
plt.title('Training and validation Loss')

plt.show()

# Zapisanie modelu
model.save('C:/Users/Kamil/Desktop/Projekt_SI/app/assets/model/flowers.h5')