import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import cv2
import os

# Загрузка данных
data = pd.read_csv('trainset.csv')
data['name'] = data['name'].str.replace('testday/', '')
image_dir = 'testday'

# Проверка баланса классов
print("Распределение классов:")
print(data['class'].value_counts())

# Параметры
img_height, img_width = 224, 224
batch_size = 32
epochs = 50

# Загрузка и предобработка изображений
images = []
labels = []

for index, row in data.iterrows():
    img_path = os.path.join(image_dir, row['name'])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_height, img_width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(row['class'])
    else:
        print(f"Изображение не найдено: {img_path}")

images = np.array(images)
labels = np.array(labels)

# Нормализация и one-hot encoding
images = images / 255.0
labels = labels - 100
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes=num_classes)

# Аугментация для редких классов (например, класс 100)
class_indices = {i: [] for i in range(num_classes)}
for idx, label in enumerate(np.argmax(labels, axis=1)):
    class_indices[label].append(idx)

aug_datagen = ImageDataGenerator(rotation_range=60,
                                 width_shift_range=0.5,
                                 height_shift_range=0.5,
                                 shear_range=0.4,
                                 zoom_range=0.5,
                                 brightness_range=[0.5, 1.5],
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='nearest')

aug_images = []
aug_labels = []
for idx in class_indices[0]:  # Класс 100
    img = images[idx]
    img = img.reshape((1, ) + img.shape)
    for _ in range(20):
        for batch in aug_datagen.flow(img, batch_size=1):
            aug_images.append(batch[0])
            aug_labels.append(labels[idx])
            break
aug_images = np.array(aug_images)
aug_labels = np.array(aug_labels)

# Объединение данных
images = np.concatenate([images, aug_images], axis=0)
labels = np.concatenate([labels, aug_labels], axis=0)

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(images,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42)

# Аугментация для тренировочных данных
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.4,
                                   height_shift_range=0.4,
                                   shear_range=0.3,
                                   zoom_range=0.4,
                                   brightness_range=[0.6, 1.4],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

train_datagen.fit(X_train)

# Создание модели
base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(img_height, img_width, 3))

base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256,
          activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
x = Dropout(0.7)(x)
x = Dense(128,
          activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
x = Dropout(0.5)(x)
x = Dense(64,
          activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.00005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Взвешивание классов
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_integers),
                                     y=y_integers)
class_weights = dict(enumerate(class_weights))

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=7,
                               restore_best_weights=True)

# Обучение
history = model.fit(train_datagen.flow(X_train, y_train,
                                       batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights,
                    callbacks=[early_stopping],
                    verbose=1)

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nТочность на тестовых данных: {test_acc:.4f}')

# Classification Report и Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказано')
plt.ylabel('Истинно')
plt.title('Матрица ошибок')
plt.savefig('confusion_matrix.png')
plt.close()

# Графики обучения
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.close()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()

# Сохранение модели
model.save('soup_classification_2505v3.keras')
