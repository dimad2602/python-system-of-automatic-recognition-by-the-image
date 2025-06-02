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
from tensorflow.keras.callbacks import EarlyStopping
import os

# Параметры
img_height, img_width = 224, 224
batch_size = 32
epochs = 20
data_dir = 'food_101'
image_dir = os.path.join(data_dir, 'images')
meta_dir = os.path.join(data_dir, 'meta')

# Загрузка классов
with open(os.path.join(meta_dir, 'classes.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]
num_classes = len(classes)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Подготовка данных с помощью ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.3,
    zoom_range=0.4,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Загрузка данных из директорий
train_generator = train_datagen.flow_from_directory(
    image_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Проверка распределения классов
print("Распределение классов в тренировочном наборе:")
print(pd.Series(train_generator.classes).value_counts())

# Для тестового набора используем test.txt
test_images = []
test_labels = []
with open(os.path.join(meta_dir, 'test.txt'), 'r') as f:
    for line in f:
        line = line.strip()
        class_name, image_id = line.split('/')
        img_path = os.path.join(image_dir, class_name, f"{image_id}.jpg")
        if os.path.exists(img_path):
            test_images.append(img_path)
            test_labels.append(class_to_idx[class_name])
        else:
            print(f"Изображение не найдено: {img_path}")

# Тестовый генератор
test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_images, 'class': [str(x) for x in test_labels]}),
    directory=None,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Создание модели
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
x = Dropout(0.7)(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Компиляция модели
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Обучение
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stopping],
    verbose=1
)

# Оценка модели
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f'\nТочность на тестовых данных: {test_acc:.4f}')

# Classification Report и Confusion Matrix
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = test_generator.classes
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=classes))

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
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
model.save('food101_classification.keras')