# результат 0.78

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#это не ошибка - все работает
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import cv2
import os

# Загрузка данных
data = pd.read_csv('trainset.csv')
data['name'] = data['name'].str.replace('testday/', '')
image_dir = 'testday'

# Параметры
img_height, img_width = 224, 224 #224 дает лучший результат
batch_size = 32
epochs = 20  # Увеличим количество эпох

# Загрузка и предобработка изображений
images = []
labels = []

for index, row in data.iterrows():
    img_path = os.path.join(image_dir, row['name'])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_height, img_width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MobileNet использует RGB
        images.append(img)
        labels.append(row['class'])
    else:
        print(f"Изображение не найдено: {img_path}")

images = np.array(images)
labels = np.array(labels)

# Нормализация и one-hot encoding
images = images / 255.0
labels = labels - 100  # Сдвиг меток (100, 101, ... -> 0, 1, ...)
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes=num_classes)

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Аугментация данных
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_datagen.fit(X_train)

# Создание модели на основе MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)


base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

#old
#base_model.trainable = False  # Замораживаем веса

inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

#new
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#old
# Компиляция модели
# model.compile(
#     optimizer=Adam(learning_rate=0.001), #0.001 дает лучший результат
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# Обучение с аугментацией
#EarlyStopping ухудшает результат
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    class_weight=class_weights,  # <--- Добавить сюда
    verbose=1
)

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nТочность на тестовых данных: {test_acc:.4f}')

# Сохранение модели
model.save('soup_classification_mobilenetv3.h5')

#confusion_matrix
#classification_report (precision, recall)
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true_classes = np.argmax(y_test, axis=1)

# print(classification_report(y_true_classes, y_pred_classes))

# cm = confusion_matrix(y_true_classes, y_pred_classes)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Предсказано')
# plt.ylabel('Истинно')
# plt.title('Матрица ошибок')
# plt.show()