# результат 0.65

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import os

data = pd.read_csv('trainset.csv')

# Удаляем префикс "testday/" из имен файлов
data['name'] = data['name'].str.replace('testday/', '')

image_dir = 'testday'

# Размеры изображения
img_height, img_width = 128, 128

# Подготовка списков с изображениями и метками классов
images = []
labels = []

for index, row in data.iterrows():
    img_path = os.path.join(image_dir, row['name'])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_height, img_width))
        images.append(img)
        labels.append(row['class'])
    else:
        print(f"Изображение не найдено: {img_path}")

# Преобразование списков в numpy массивы
if images:
    images = np.array(images)
    labels = np.array(labels)
else:
    raise ValueError(
        "Нет загруженных изображений. Проверьте путь к файлам и данные в CSV.")

# Нормализация изображений
images = images / 255.0

# Проверка уникальных классов
# unique_labels = np.unique(labels)
# print(f"Уникальные метки классов: {unique_labels}")
# num_classes = len(unique_labels)
# print(f"Количество уникальных классов: {num_classes}")

# Сдвиг меток, чтобы они начинались с 0
labels = labels - 100

# Проверка уникальных классов после сдвига
unique_labels = np.unique(labels)
print(f"Уникальные метки классов после сдвига: {unique_labels}")

# Преобразование меток в one-hot encoding
num_classes = len(unique_labels)
labels = to_categorical(labels, num_classes=num_classes)

# Проверка, что форма labels теперь (samples, num_classes)
print(f"Форма labels после to_categorical: {labels.shape}")

model = Sequential([
    Conv2D(32, (3, 3),
           activation='relu',
           input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(images,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42)

# Проверка формы после one-hot кодирования
print(f"Форма X_train: {X_train.shape}")
print(f"Форма y_train: {y_train.shape}")
print(f"Форма X_test: {X_test.shape}")
print(f"Форма y_test: {y_test.shape}")

# Убедимся, что y_train и y_test имеют правильную форму
if len(y_train.shape) == 3:
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

print(f"Форма y_train после squeeze: {y_train.shape}")
print(f"Форма y_test после squeeze: {y_test.shape}")


history = model.fit(X_train,
                    y_train,
                    epochs=20, #10
                    batch_size=32,
                    validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nТочность на тестовых данных: {test_acc:.2f}')


model.save('soup_classification_model.h5')
model.save('soup_classification_model.keras')
