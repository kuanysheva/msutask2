import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from pathlib import Path
from PIL import Image
import csv
import os
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# Функция для загрузки данных из папки
def load_images_and_labels(directory: Path):
    frames, labels, coords = [], [], []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                img = Image.open(os.path.join(root, file)).resize((224, 224))
                frames.append(np.array(img, dtype=np.uint8))
            elif file == 'labels.csv':
                with open(os.path.join(root, file), newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)
                    for row in reader:
                        try:
                            labels.append(int(row[1]) if row[1] else 0)
                            coords.append([int(row[2]) if row[2] else 0, int(row[3]) if row[3] else 0])
                        except ValueError:
                            print(f"Invalid value in row: {row}")
    return np.array(frames), np.array(labels), np.array(coords)


# Создание модели
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = Dropout(0.5)(Flatten()(base_model.output))
    model = Model(inputs=base_model.input, outputs=[Dense(1, activation='sigmoid', name='classification')(x),
                                                    Dense(2, activation='linear', name='regression')(x)])
    return model


# Компиляция модели
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss={'classification': 'binary_crossentropy', 'regression': 'mse'},
              loss_weights={'classification': 1.0, 'regression': 0.5}, metrics={'classification': 'accuracy'})


# Тренировка модели
def train_model(model, train_data, val_data, epochs=10, batch_size=32):
    train_frames, train_labels, train_coords = train_data
    val_frames, val_labels, val_coords = val_data
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_frames, {'classification': train_labels, 'regression': train_coords})).map(
        lambda x, y: (tf.image.resize(tf.cast(x, tf.float32) / 255.0, (224, 224)), y)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_frames, {'classification': val_labels, 'regression': val_coords})).map(
        lambda x, y: (tf.image.resize(tf.cast(x, tf.float32) / 255.0, (224, 224)), y)).batch(batch_size)

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    # Построение графиков
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['classification_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    plt.savefig('training_results.png')
    plt.show()

    return history


# Функция для расчета Simple Ball Tracking Accuracy (SiBaTrAcc)
def calculate_sibatracc(predictions, ground_truth, e1=5, e2=5, step=8, alpha=1.5):
    def P_error(code_pr, x_pr, y_pr, code_gt, x_gt, y_gt):
        if code_gt != 0 and code_pr == 0:
            return e1
        elif code_gt == 0 and code_pr != 0:
            return e2
        else:
            distance = np.sqrt((x_gt - x_pr) ** 2 + (y_gt - y_pr) ** 2)
            return min(5, distance / step) ** alpha

    total_error = 0
    N = len(predictions)
    for pred, gt in zip(predictions, ground_truth):
        code_pr, x_pr, y_pr = pred
        code_gt, x_gt, y_gt = gt
        total_error += P_error(code_pr, x_pr, y_pr, code_gt, x_gt, y_gt)

    SiBaTrAcc = 1 - (total_error / (5 * N))
    return SiBaTrAcc


# Подготовка данных для оценки
def prepare_evaluation_data(test_frames, test_labels, test_coords, model):
    test_predictions = []
    for frame in test_frames:
        frame_preprocessed = tf.image.resize(tf.cast(frame, tf.float32) / 255.0, (224, 224))
        pred_classification, pred_regression = model.predict(tf.expand_dims(frame_preprocessed, axis=0))
        code_pr = 1 if pred_classification[0] > 0.5 else 0
        x_pr, y_pr = pred_regression[0]
        test_predictions.append((code_pr, x_pr, y_pr))
    ground_truth = [(code, x, y) for code, (x, y) in zip(test_labels, test_coords)]
    return test_predictions, ground_truth


# Создание PDF отчета
def create_pdf_report(history, sibatracc_score):
    pdf = canvas.Canvas("report.pdf", pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica", 12)
    pdf.drawString(30, height - 30, "Training and Evaluation Report")
    pdf.drawString(30, height - 50, f"Simple Ball Tracking Accuracy (SiBaTrAcc): {sibatracc_score:.4f}")

    pdf.drawImage('training_results.png', 30, height - 300, width=550, height=250)

    pdf.save()


# Загрузка данных
dataset_path = Path("D:/MSU/archive")
train_data = load_images_and_labels(dataset_path / "train")
val_data = load_images_and_labels(dataset_path / "test")

# Тренировка и оценка модели
history = train_model(model, train_data, val_data)

# Оценка на тестовом наборе с использованием метрики SiBaTrAcc
test_predictions, ground_truth = prepare_evaluation_data(val_data[0], val_data[1], val_data[2], model)
sibatracc_score = calculate_sibatracc(test_predictions, ground_truth)
print("Simple Ball Tracking Accuracy (SiBaTrAcc):", sibatracc_score)

# Создание PDF отчета
create_pdf_report(history, sibatracc_score)
