import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
import os

# Загрузка данных
data = pd.read_csv('trainset.csv')
image_dir = 'testday'

# Параметры
IMG_SIZE = 224
CLASSES = [f"Суп {i}" for i in range(100, 106)]

# Загрузка модели
model = load_model("soup_classification_25053.keras")

# Подготовка GUI
root = Tk()
root.title("Распознавание супа по камере")

# Камера
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: камера не доступна")
    exit()

panel = Label(root)
panel.pack()

result_label = Label(root, text="Нажмите кнопку для распознавания", font=("Arial", 14))
result_label.pack(pady=10)

def show_frame():
    ret, frame = cap.read()
    if not ret:
        result_label.config(text="Ошибка считывания с камеры")
        root.after(1000, show_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.configure(image=imgtk)
    root.after(10, show_frame)

def capture_and_predict():
    ret, frame = cap.read()
    if not ret:
        result_label.config(text="Ошибка камеры")
        return

    # Сохранение входного изображения
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # cv2.imwrite(f"input_{timestamp}.jpg", frame)

    # Предобработка
    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Предсказание
    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    class_name = CLASSES[class_id]
    confidence = np.max(prediction)

    # Вывод всех вероятностей
    result_text = f"Суп: {class_name} ({confidence:.2f})\n"
    for i, prob in enumerate(prediction[0]):
        result_text += f"{CLASSES[i]}: {prob:.2f}\n"

    # Обновление UI
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    panel.configure(image=imgtk)
    panel.imgtk = imgtk
    result_label.config(text=result_text)

    # Сохранение в orders.csv
    with open("orders.csv", "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{class_name},{confidence:.2f}\n")

    # Логирование
    # with open("predictions.log", "a") as f:
    #     f.write(f"{timestamp}: {prediction}\n")

def test_on_train_image():
    img_path = os.path.join(image_dir, data['name'].iloc[0])
    frame = cv2.imread(img_path)
    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    class_name = CLASSES[class_id]
    confidence = np.max(prediction)
    result_text = f"Суп: {class_name} ({confidence:.2f})\n"
    for i, prob in enumerate(prediction[0]):
        result_text += f"{CLASSES[i]}: {prob:.2f}\n"
    result_label.config(text=result_text)

btn = Button(root, text="Определить состав заказа", command=capture_and_predict, font=("Arial", 12))
btn.pack(pady=5)
btn_test = Button(root, text="Тест на тренировочном изображении", command=test_on_train_image, font=("Arial", 12))
btn_test.pack(pady=5)

# Запуск
show_frame()
root.mainloop()

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()