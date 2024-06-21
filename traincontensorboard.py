import os
import time
from glob import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
def accuracy(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    correct = (y_pred == y_true).float().sum()
    total = y_true.numel()
    acc = correct / total
    return acc.item()

def intersection_over_union(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    iou = intersection / union
    return iou.item()

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Calcular y guardar accuracy y IoU
        y_pred = (y_pred > 0.5).float()
        train_acc = accuracy(y_pred, y)
        train_iou = intersection_over_union(y_pred, y)

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss, train_acc, train_iou

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    
    all_y_pred = []
    all_y_true = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            # Calcular y guardar accuracy y IoU para la validación
            y_pred = (y_pred > 0.5).float()
            valid_acc = accuracy(y_pred, y)
            valid_iou = intersection_over_union(y_pred, y)
 # Agregar las predicciones y etiquetas verdaderas a las listas
            all_y_pred.append(y_pred.cpu().numpy())
            all_y_true.append(y.cpu().numpy())

        # Concatenar todas las predicciones y etiquetas verdaderas
        all_y_pred = np.concatenate(all_y_pred)
        all_y_true = np.concatenate(all_y_true)
        epoch_loss = epoch_loss/len(loader)

    return epoch_loss, valid_acc, valid_iou,all_y_pred, all_y_true

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files2")

    """ Load dataset """
    train_x = sorted(glob(r"D:\ruebastesis\ruebaidiot\new_data2\train\image\*"))
    train_y = sorted(glob(r"D:\ruebastesis\ruebaidiot\new_data2\train\mask\*"))

    valid_x = sorted(glob(r"D:\ruebastesis\ruebaidiot\new_data2\test\image\*"))
    valid_y = sorted(glob(r"D:\ruebastesis\ruebaidiot\new_data2\test\mask\*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 500
    lr = 1e-4
    checkpoint_path = "files2/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda')   ## GTX 1060 6GB
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")
    # Crea un objeto SummaryWriter para TensorBoard
    writer = SummaryWriter('logs')
    # Crea un DataFrame vacío para almacenar los datos de entrenamiento
    columns = ['Epoch', 'Epoch Time (m)', 'Train Loss', 'Valid Loss', 'Train Acc', 'Train IoU', 'Valid Acc', 'Valid IoU',"TN","FP","FN","TP"]
    df = pd.DataFrame(columns=columns)
    # Define el nombre del archivo Excel
    excel_file = r"files2\training_log.xlsx"

    # Verifica si el archivo Excel ya existe
    if os.path.exists(excel_file):
        # Si el archivo existe, carga los datos existentes en un DataFrame
        df = pd.read_excel(excel_file)
        # Obtiene la última época registrada en el archivo Excel
        last_epoch = df['Epoch'].max()
        # Define la época desde la que deseas iniciar (1 más que la última época)
        start_epoch = last_epoch
        print(f"Continuando desde la época {start_epoch}")
    else:
        # Si el archivo no existe, inicia desde la época 1
        start_epoch = 0
        print("No se encontró ningún archivo Excel. Comenzando desde la época 1.")

    # Ahora puedes usar start_epoch para continuar el entrenamiento desde donde se quedó

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        train_loss, train_acc, train_iou = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_acc, valid_iou ,all_y_pred, all_y_true= evaluate(model, valid_loader, loss_fn, device)
        # Convertir a matrices binarias (0 o 1)
        all_y_true_bin = (all_y_true > 0.5).astype(int)
        all_y_pred_bin = (all_y_pred > 0.5).astype(int)
        """ Actualizar el scheduler con la pérdida de validación """
        scheduler.step(valid_loss)

        """ Imprimir la tasa de aprendizaje actual """
        print("Tasa de aprendizaje actual:", optimizer.param_groups[0]['lr'])

        """ Guardar el modelo """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Train IoU: {train_iou:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f} | Valid Acc: {valid_acc:.3f} | Valid IoU: {valid_iou:.3f}\n'
        print(data_str)

        # Crear un arreglo con los valores de IoU de entrenamiento y validación
        iou_values = np.array([train_iou, valid_iou])
        loss_values = np.array([train_loss, valid_loss])
        acc_values = np.array([train_acc, valid_acc])
        # Agregar el histograma del IoU de entrenamiento y validación a TensorBoard
        


        # Agrega las métricas a TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Loss/valid', valid_loss, epoch+1)
        writer.add_scalar('Accuracy/train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch+1)
        writer.add_scalar('IoU/train', train_iou, epoch+1)
        writer.add_scalar('IoU/valid', valid_iou, epoch+1)
        #writer.add_histogram('IoU/Train', train_iou, epoch)
        writer.add_histogram('IoU/Train-Valid', iou_values, epoch+1)
        writer.add_histogram('Loss/Train-Valid', loss_values, epoch+1)
        writer.add_histogram('Accuracy/Train-Valid', acc_values, epoch+1)
        # Calcular la matriz de confusión
        conf_matrix = confusion_matrix(all_y_true_bin.flatten(), all_y_pred_bin.flatten())
        print(all_y_true_bin.flatten())
        print("Matriz de confusión:")
        print(conf_matrix)
        # Calcular el informe de clasificación
        class_report = classification_report(all_y_true_bin.flatten(), all_y_pred_bin.flatten())
        # # Imprimir y guardar el informe de clasificación
        # print("Informe de Clasificación:")
        print(class_report)
        # Agregar el histograma de la matriz de confusión a TensorBoard
        # Agregar el histograma de la matriz de confusión a TensorBoard
        writer.add_histogram('Confusion Matrix', conf_matrix, epoch+1) 
        # Agrega los datos al DataFrame
        df.loc[epoch] = [epoch+1, epoch_mins, train_loss, valid_loss, train_acc, train_iou, valid_acc, valid_iou,conf_matrix[0,0],conf_matrix[0,1],conf_matrix[1,0],conf_matrix[1,1]]
         # Guarda el DataFrame en un archivo Excel después de cada época
        excel_file = f"files2/training_log.xlsx"
        df.to_excel(excel_file, index=False)       
    # Crear el gráfico de la matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.title('Matriz de Confusión')
        plt.savefig(f'files2/confusion_matrix_epoch_{epoch+1}.png')
        plt.close()
        #------------------------------------------------
        # # Calcular la matriz de correlación
        # corr_matrix = df.corr()

        # # Crear el gráfico de mapa de calor
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title('Correlation Matrix of Performance Metrics')
        # plt.xlabel('Metrics')
        # plt.ylabel('Metrics')

        # # Guardar el gráfico como un archivo PNG
        # #plt.savefig('correlation_matrix.png')
        #-----------------------------------------------------------
        
    # Cierra el SummaryWriter al finalizar
    writer.close()

#     [[TN, FP]
#  [FN, TP]]
# TN (True Negative): La cantidad de ejemplos que el modelo ha clasificado correctamente como negativos.

# FP (False Positive): La cantidad de ejemplos que el modelo ha clasificado incorrectamente como positivos cuando en realidad son negativos.

# FN (False Negative): La cantidad de ejemplos que el modelo ha clasificado incorrectamente como negativos cuando en realidad son positivos.

# TP (True Positive): La cantidad de ejemplos que el modelo ha clasificado correctamente como positivos.