import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os
import torch
import timm
from tensorflow import keras

model1 = keras.models.load_model("inceptionV3.h5")
model2 = keras.models.load_model("ConvNext_Model.keras")
model3 = keras.models.load_model("densenet121_Model.h5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model4 = timm.create_model('maxvit_tiny_rw_224', pretrained=False, num_classes=len(os.listdir("flowers")))
model4.load_state_dict(torch.load("maxvit_flower_classifier.pth", map_location=device))
model4.to(device)
model4.eval()

model5 = timm.create_model('levit_128s', pretrained=False, num_classes=len(os.listdir("flowers")))
model5.load_state_dict(torch.load("levit_flower_classifier.pth", map_location=device))
model5.to(device)
model5.eval()

model6 = timm.create_model('mobilevitv2_050', pretrained=False, num_classes=len(os.listdir("flowers")))
model6.load_state_dict(torch.load("mobilevit_flower_classifier.pth", map_location=device))
model6.to(device)
model6.eval()

image_folder = "flowers"
class_names = [name for name in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, name))]
class_names.sort()

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            inputs = {
                model1: np.expand_dims(keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(file_path, target_size=(299, 299))), axis=0) / 255.0,
                model2: np.expand_dims(keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(file_path, target_size=(224, 224))), axis=0) / 255.0,
                model3: np.expand_dims(keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(file_path, target_size=(299, 299))), axis=0) / 255.0,
            }

            preds = [
                (model1.predict(inputs[model1]), result_label1, progressbar1, confidence_label1),
                (model2.predict(inputs[model2]), result_label2, progressbar2, confidence_label2),
                (model3.predict(inputs[model3]), result_label3, progressbar3, confidence_label3),
            ]

            model_names = ["InceptionV3", "ConvNeXt", "DenseNet121"]

            for i, (pred, label, bar, conf_label) in enumerate(preds):
                idx = np.argmax(pred)
                confidence = np.max(pred) * 100
                label.configure(text=f"{model_names[i]}: {class_names[idx]}")
                bar.set(confidence / 100)
                conf_label.configure(text=f"%{confidence:.2f}")
                color = "#4ecca3" if confidence > 80 else "#f9a825" if confidence > 50 else "#f44336"
                bar.configure(progress_color=color)
                label.configure(text_color=color)

            image = Image.open(file_path).resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.transpose(image_array, (2, 0, 1))
            image_tensor = torch.tensor(image_array).float().unsqueeze(0).to(device)

            with torch.no_grad():
                output4 = model4(image_tensor)
                _, idx4 = torch.max(output4, 1)
                conf4 = torch.nn.functional.softmax(output4, dim=1).max().item() * 100
                result_label4.configure(text=f"MaxViT: {class_names[idx4.item()]}")
                progressbar4.set(conf4 / 100)
                confidence_label4.configure(text=f"%{conf4:.2f}")
                color4 = "#4ecca3" if conf4 > 80 else "#f9a825" if conf4 > 50 else "#f44336"
                progressbar4.configure(progress_color=color4)
                result_label4.configure(text_color=color4)

            with torch.no_grad():
                output5 = model5(image_tensor)
                _, idx5 = torch.max(output5, 1)
                conf5 = torch.nn.functional.softmax(output5, dim=1).max().item() * 100
                result_label5.configure(text=f"LeViT: {class_names[idx5.item()]}")
                progressbar5.set(conf5 / 100)
                confidence_label5.configure(text=f"%{conf5:.2f}")
                color5 = "#4ecca3" if conf5 > 80 else "#f9a825" if conf5 > 50 else "#f44336"
                progressbar5.configure(progress_color=color5)
                result_label5.configure(text_color=color5)

            with torch.no_grad():
                output6 = model6(image_tensor)
                _, idx6 = torch.max(output6, 1)
                conf6 = torch.nn.functional.softmax(output6, dim=1).max().item() * 100
                result_label6.configure(text=f"MobileViT: {class_names[idx6.item()]}")
                progressbar6.set(conf6 / 100)
                confidence_label6.configure(text=f"%{conf6:.2f}")
                color6 = "#4ecca3" if conf6 > 80 else "#f9a825" if conf6 > 50 else "#f44336"
                progressbar6.configure(progress_color=color6)
                result_label6.configure(text_color=color6)

            img = Image.open(file_path)
            img = ImageOps.fit(img, (300, 300), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            panel.configure(image=img, text="")
            panel.image = img

        except Exception as e:
            messagebox.showerror("Hata", f"Görsel işlenirken bir hata oluştu:\n{str(e)}", icon="error")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")
root = ctk.CTk()
root.title("Çiçek Sınıflandırma Uygulaması")

window_width = 1100
window_height = 900
x = (root.winfo_screenwidth() // 2) - (window_width // 2)
y = (root.winfo_screenheight() // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.resizable(False, False)
root.configure(fg_color="#2b2b2b")

header_frame = ctk.CTkFrame(root, fg_color="transparent")
header_frame.pack(pady=10)
title_label = ctk.CTkLabel(header_frame, text="Çiçek Sınıflandırma", font=("Arial", 24, "bold"), text_color="#4ecca3")
title_label.pack()

main_frame = ctk.CTkFrame(root, fg_color="transparent")
main_frame.pack(pady=5)
panel = ctk.CTkLabel(main_frame, text="Resim Seçin", width=300, height=300, fg_color="#3a3a3a", corner_radius=15, font=("Arial", 13), text_color="#aaaaaa")
panel.pack(pady=10)

button_frame = ctk.CTkFrame(root, fg_color="transparent")
button_frame.pack(pady=5)
btn = ctk.CTkButton(button_frame, text="Resim Seç", command=predict_image, corner_radius=10, height=40, width=180, font=("Arial", 13, "bold"), fg_color="#4ecca3", hover_color="#3aa789", text_color="#ffffff")
btn.pack(pady=5)

results_frame = ctk.CTkFrame(root, fg_color="transparent")
results_frame.pack(pady=10)

def create_result_section(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(side="left", padx=10)
    label = ctk.CTkLabel(frame, text="", font=("Arial", 16, "bold"))
    label.pack(pady=5)
    bar = ctk.CTkProgressBar(frame, width=200, height=12, corner_radius=8, fg_color="#3a3a3a")
    bar.set(0)
    bar.pack(pady=3)
    confidence = ctk.CTkLabel(frame, text="%0", font=("Arial", 13), text_color="#aaaaaa")
    confidence.pack(pady=3)
    return label, bar, confidence

result_label1, progressbar1, confidence_label1 = create_result_section(results_frame)
result_label2, progressbar2, confidence_label2 = create_result_section(results_frame)
result_label3, progressbar3, confidence_label3 = create_result_section(results_frame)

pytorch_frame = ctk.CTkFrame(root, fg_color="transparent")
pytorch_frame.pack(pady=10)
result_label4, progressbar4, confidence_label4 = create_result_section(pytorch_frame)
result_label5, progressbar5, confidence_label5 = create_result_section(pytorch_frame)
result_label6, progressbar6, confidence_label6 = create_result_section(pytorch_frame)

root.mainloop()