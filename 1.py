import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import os

# Загрузка данных из CSV
data = pd.read_csv('data.csv')
image_dir = 'path/to/images_folder'

# Размеры изображения
img_height, img_width = 128, 128

# Подготовка списков с изображениями и метками классов
images = []
labels = []

for index, row in data.iterrows():
    img_path = os.path.join(image_dir, row['filename'])
    # Проверка на существование изображения
    if os.path.exists(img_path):
        # Чтение и изменение размера изображения
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_height, img_width))
        images.append(img)
        labels.append(row['class'])

# Преобразование списков в numpy массивы
images = np.array(images)
labels = np.array(labels)

# Нормализация изображений
images = images / 255.0

# Преобразование меток в one-hot encoding
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)
