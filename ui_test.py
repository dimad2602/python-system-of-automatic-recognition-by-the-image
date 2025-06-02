import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk

import pandas as pd

data = pd.read_csv('trainset.csv')
print(data['class'].value_counts())

# Параметры
#Разобатся с размерами
IMG_SIZE = 224
CLASSES = [f"Суп {i}" for i in range(100, 106)]  # или свои названия супов

# Загрузка модели
model = load_model("soup_classification_25053.keras") #soup_classification_mobilenetv4.h5

# Подготовка GUI
root = Tk()
root.title("Распознавание супа по камере")

# Камера
cap = cv2.VideoCapture(0)

panel = Label(root)
panel.pack()

result_label = Label(root,
                     text="Нажмите кнопку для распознавания",
                     font=("Arial", 14))
result_label.pack(pady=10)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: камера не доступна")
    exit()


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

    frame = frame[50:450, 50:450]

    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image / 255.0)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    class_name = CLASSES[class_id]

    # Показываем картинку и результат
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    panel.configure(image=imgtk)
    panel.imgtk = imgtk

    confidence = np.max(prediction)
    result_label.config(text=f"Суп: {class_name} ({confidence:.2f})")


btn = Button(root,
             text="Определить состав заказа",
             command=capture_and_predict,
             font=("Arial", 12))
btn.pack(pady=5)

# Запуск потока
show_frame()
root.mainloop()

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

#Сохранить в заказ
# with open("orders.csv", "a", encoding="utf-8") as f:
#     f.write(f"{class_name},{confidence:.2f}\n")
