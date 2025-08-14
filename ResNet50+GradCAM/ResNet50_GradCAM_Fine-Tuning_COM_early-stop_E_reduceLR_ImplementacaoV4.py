# ===================== BIBLIOTECAS =====================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
from tkinter import filedialog, Tk

# === Constantes ===
IMG_SIZE = (224, 224)
NUM_CLASSES = 6
CLASS_NAMES = ["Normal", "HP", "TA.HG", "TA.LG", "TVA.HG", "TVA.LG"]
train_data, train_labels = [], []
test_data, test_labels = [], []
gradcam_refs = {i: [] for i in range(NUM_CLASSES)}

# === Upload de imagens via Tkinter ===
def upload_images_for_class(stage, class_index):
    print(f"\n Selecione imagens para {stage} da classe: {CLASS_NAMES[class_index]}")
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title=f"{stage} - {CLASS_NAMES[class_index]}",
                                             filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    images, raw_images = [], []
    for file_path in file_paths:
        img = load_img(file_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        raw_images.append(np.array(img))
        img_array = preprocess_input(img_array)
        images.append(img_array)
    labels = [class_index] * len(images)
    if stage == "TREINAMENTO":
        gradcam_refs[class_index].extend(raw_images)
    return images, labels

# === Coletar dados de treino ===
# Itera sobre cada classe de 0 até NUM_CLASSES-1, permitindo carregar imagens para todas as classes
# Função personalizada para fazer upload (ou carregamento local) das imagens da pasta "TREINAMENTO" referentes à classe 'i'.
# Retorna: imgs (lista de arrays de imagens) e lbls (lista de rótulos inteiros correspondentes).
for i in range(NUM_CLASSES):
    imgs, lbls = upload_images_for_class("TREINAMENTO", i)
    train_data.extend(imgs)                                          # Adiciona as imagens carregadas na lista principal de dados de treino.
    train_labels.extend(lbls)                                        # Adiciona os rótulos correspondentes na lista principal de rótulos.
# Converte a lista de imagens em um array NumPy para processamento vetorizado, exigido pelo Keras/TensorFlow.
train_data = np.array(train_data)
# Converte os rótulos de inteiros para codificação one-hot com 'NUM_CLASSES' colunas (saída compatível com softmax).
train_labels = to_categorical(train_labels, NUM_CLASSES)

# === Separar treino e validação manualmente ===
# Divide o conjunto em treino (80%) e validação (20%).
# test_size=0.2 → 20% para validação.
# stratify=train_labels → mantém a proporção das classes.
# random_state=42 → garante reprodutibilidade da divisão.
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# === Data Augmentation ===
# Aumento de dados (gera variações artificiais das imagens de treino para evitar overfitting).
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(
    rotation_range=20,                # Rotaciona imagens aleatoriamente entre -20° e +20°.
    width_shift_range=0.1,            # Desloca horizontalmente até 10% da largura da imagem.
    height_shift_range=0.1,           # Desloca verticalmente até 10% da altura da imagem.
    zoom_range=0.1,                   # Aplica zoom aleatório de até 10%.
    horizontal_flip=True              # Inverte a imagem horizontalmente de forma aleatória.
)
# Esses métodos aumentam a diversidade dos dados de treino, melhorando a capacidade de generalização.
val_gen = ImageDataGenerator()                    # Não aplica aumentos nos dados de validação (mantém as imagens originais para avaliação justa).
# Gera lotes de treino com 16 imagens cada (batch_size=16 → equilíbrio entre desempenho e estabilidade da atualização de pesos).
train_generator = train_gen.flow(train_data, train_labels, batch_size=16)
# Gera lotes de validação com o mesmo tamanho.
val_generator = val_gen.flow(val_data, val_labels, batch_size=16)

# === Callbacks ===
# Para o treino se a 'val_loss' não melhorar por 5 épocas consecutivas.
# restore_best_weights=True → restaura o modelo para o ponto de menor 'val_loss'.
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Reduz a taxa de aprendizado pela metade (factor=0.5) se 'val_loss' não melhorar por 3 épocas.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# === MODELO ResNet50 ===
# Carrega a ResNet50 pré-treinada no ImageNet.
# input_shape=(224,224,3) → imagens RGB com 224x224 pixels.
# include_top=False → permite adicionar camadas densas personalizadas.
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Converte mapas de características 2D em um vetor 1D por média global, reduzindo parâmetros e overfitting.
x = GlobalAveragePooling2D()(base_model.output)
# Dropout de 50% → desativa aleatoriamente metade das unidades para regularizar o modelo.
x = Dropout(0.5)(x)
# Camada densa com 512 neurônios, ativação ReLU e regularização L2 com peso 0.0001.
x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
# Outro Dropout de 50% para reforçar a regularização.
x = Dropout(0.5)(x)
# Camada de saída com 'NUM_CLASSES' neurônios (uma por classe), ativação Softmax para classificação multiclasse.
output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(1e-4))(x)
# Define o modelo final unindo entrada da ResNet50 com a nova cabeça de classificação.
model = Model(inputs=base_model.input, outputs=output)
# Congela as camadas convolucionais para preservar pesos pré-treinados durante a primeira fase de treino.
for layer in base_model.layers:
    layer.trainable = False
# Otimizador Adam com taxa de aprendizado inicial 0.001.
# Função de perda categorical_crossentropy (multiclasse).
# Métrica principal: acurácia.
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
print("\n Treinamento inicial das camadas densas...")
# Treina apenas as camadas densas por até x épocas (early_stop provavelmente interrompe antes).
history = model.fit(train_generator, validation_data=val_generator, epochs=80, callbacks=[early_stop, reduce_lr])

# ==== FINE-TUNING (TODAS AS CAMADAS) ====
# Descongela todas as camadas para ajuste fino.
for layer in base_model.layers:
    layer.trainable = True
# Nova taxa de aprendizado menor (0.00001) para evitar destruir pesos já bem ajustados.
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print("\n Fine-tuning de todas as camadas do modelo...")
# Treina todas as camadas com taxa reduzida por até x épocas.
history_ft = model.fit(train_generator, validation_data=val_generator, epochs=80, callbacks=[early_stop, reduce_lr])

# === Upload de dados de teste ===
# Carrega imagens e rótulos do conjunto de teste, classe por classe.
for i in range(NUM_CLASSES):
    imgs, lbls = upload_images_for_class("TESTE", i)
    test_data.extend(imgs)
    test_labels.extend(lbls)
# Converte lista de imagens de teste para array NumPy.
test_data = np.array(test_data)
# Converte rótulos de teste para codificação.
test_labels_cat = to_categorical(test_labels, NUM_CLASSES)

# === Predição ===
# Gera probabilidades previstas para cada classe de cada imagem de teste.
pred_probs = model.predict(test_data)
# Converte probabilidades para rótulos previstos (classe com maior probabilidade).
pred_labels = np.argmax(pred_probs, axis=1)
# Converte lista de rótulos reais para array NumPy (facilita métricas de avaliação).
true_labels = np.array(test_labels)

# === Métricas ===
acc = accuracy_score(true_labels, pred_labels)
bal_acc = balanced_accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')
kappa = cohen_kappa_score(true_labels, pred_labels)

cm = confusion_matrix(true_labels, pred_labels)
specificity = []
for i in range(NUM_CLASSES):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity.append(tn / (tn + fp))

# === Gráficos de desempenho separados ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Loss Treino')
plt.plot(history.history['val_loss'], label='Loss Validação')
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.title('Treinamento Inicial - Camadas Densas')
plt.xlabel('Epochs')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history_ft.history['loss'], label='Loss Treino')
plt.plot(history_ft.history['val_loss'], label='Loss Validação')
plt.plot(history_ft.history['accuracy'], label='Acurácia Treino')
plt.plot(history_ft.history['val_accuracy'], label='Acurácia Validação')
plt.title('Fine-Tuning - Todas as Camadas')
plt.xlabel('Epochs')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Exibir métricas ===
print("\n MÉTRICAS DE DESEMPENHO - CNN ResNet50")
print(f"Acurácia: {acc:.4f}")
print(f"Acurácia Balanceada: {bal_acc:.4f}")
print(f"Precisão (macro): {prec:.4f}")
print(f"Revocação (Sensibilidade): {recall:.4f}")
print(f"F1-Score (macro): {f1:.4f}")
print(f"Índice Kappa: {kappa:.4f}")
print(f"Especificidade (média): {np.mean(specificity):.4f}")

# === Matriz de Confusão ===
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# === Curvas ROC ===
def plot_roc_curve_binary(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, auc(fpr, tpr)

plt.figure(figsize=(8, 6))
colors = ['darkorange', 'green', 'blue', 'purple', 'red']
for i in range(1, NUM_CLASSES):
    bin_idx = np.isin(test_labels, [0, i])
    bin_true = np.array([1 if lbl == i else 0 for lbl in np.array(test_labels)[bin_idx]])
    bin_score = pred_probs[bin_idx, i]
    fpr, tpr, roc_auc = plot_roc_curve_binary(bin_true, bin_score)
    plt.plot(fpr, tpr, label=f"{CLASS_NAMES[0]} vs {CLASS_NAMES[i]} (AUC = {roc_auc:.4f})", color=colors[i-1])

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Curvas ROC Binárias")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Grad-CAM ===
#Função para gerar o mapa de calor (heatmap) do Grad-CAM para uma imagem específica
#img_array: imagem já pré-processada em formato NumPy pronta para entrada no modelo.
#model: rede neural treinada utilizada para gerar as ativações.
#last_conv_layer_name: nome da última camada convolucional (se não informado, será buscado automaticamente).
#pred_index: índice da classe-alvo para gerar o Grad-CAM (se não informado, será a classe de maior probabilidade).
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    #Busca a última camada convolucional do modelo caso last_conv_layer_name não tenha sido passado.
    #Grad-CAM precisa das ativações da última camada convolucional antes da parte totalmente conectada (Dense), pois são essas ativações que têm mapeamento espacial.
    #Percorre as camadas do modelo de trás para frente (reversed(model.layers)) e seleciona a primeira instância de Conv2D
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    #Cria um novo modelo que retorna duas saídas: As ativações da última camada convolucional e a predição final do modelo
    #Necessário para calcular os gradientes entre a classe alvo e o mapa de ativação.
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    #tf.GradientTape(): Permite calcular automaticamente os gradientes das saídas em relação às entradas.
    #conv_outputs: mapa de ativação da última camada convolucional.
    #predictions: vetor de probabilidades para cada classe (dimensão igual a NUM_CLASSES).
    #tf.argmax(predictions[0]): seleciona a classe de maior probabilidade quando pred_index não é informado.
    #class_output: escalar que representa a probabilidade da classe escolhida — é sobre ele que calcularemos os gradientes.
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    #Calcula o gradiente da saída (class_output) em relação às ativações da última camada convolucional (conv_outputs).
    #(IMPORTANTE)-Esses gradientes indicam quais regiões do mapa de ativação contribuem mais para a decisão do modelo.
    grads = tape.gradient(class_output, conv_outputs)
    #Faz uma média global por canal dos gradientes.
    #axis=(0, 1, 2) significa que está promediando ao longo de altura, largura e batch.
    #O resultado é um vetor com um valor para cada canal de ativação, representando a importância desse canal para a classe.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    #Remove a dimensão de batch para trabalhar apenas com o mapa de ativação da imagem única.
    conv_outputs = conv_outputs[0]
    #Multiplica cada canal do mapa de ativação pelo seu peso de importância (pooled_grads) e soma o resultado (produto escalar sobre canais).
    #Isso gera um mapa 2D representando a importância espacial das regiões
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    #Remove dimensões unitárias, garantindo que o heatmap seja apenas uma matriz 2D (altura × largura)
    heatmap = tf.squeeze(heatmap)
    #tf.maximum(heatmap, 0): aplica ReLU para manter apenas valores positivos (regiões que contribuem positivamente).
    #Normaliza dividindo pelo valor máximo para que fique no intervalo [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    #Converte o tensor para NumPy para uso posterior (ex.: manipulação no OpenCV).
    return heatmap.numpy()

#Define o diretório onde serão salvos os resultados (~/Downloads/gradcam_results) e cria a pasta caso não exista.
def save_gradcam_images():
    download_path = Path.home() / 'Downloads' / 'gradcam_results'
    download_path.mkdir(parents=True, exist_ok=True)
    #Itera sobre cada classe (class_index) e suas imagens (images).
    #images[:30]: hiperparâmetro fixo = 30 imagens por classe para salvar.
    for class_index, images in gradcam_refs.items():
        for i, raw_img in enumerate(images[:30]):
            #img_to_array: converte imagem PIL para array NumPy.
            #preprocess_input: aplica a normalização adequada ao backbone usado (ex.: ResNet, NASNet, etc.).
            #np.expand_dims: adiciona dimensão de batch ((1, altura, largura, canais)).
            img_array = preprocess_input(img_to_array(raw_img))
            img_array = np.expand_dims(img_array, axis=0)
            #Gera o heatmap e redimensiona para o tamanho original da imagem (IMG_SIZE é um hiperparâmetro global, ex.: (224, 224)).
            heatmap = make_gradcam_heatmap(img_array, model)
            heatmap = cv2.resize(heatmap, IMG_SIZE)
            #Converte valores de [0,1] para [0,255] em formato uint8 para exibição.
            heatmap = np.uint8(255 * heatmap)
            #Aplica coloração de mapa térmico (azul → verde → amarelo → vermelho) para destacar regiões quentes.
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            #Fusão de imagens: Imagem original com peso 0.6 e Heatmap colorido com peso 0.4 sendo 0 = deslocamento de brilho
            #Isso gera a sobreposição Grad-CAM visualmente interpretável.
            overlay = cv2.addWeighted(np.array(raw_img), 0.6, heatmap_color, 0.4, 0)
            #Converte o overlay para imagem PIL e salva no disco com nome "class_<classe>_img_<n>.jpg".
            output_image = Image.fromarray(overlay.astype(np.uint8))
            output_image.save(download_path / f'class_{class_index}_img_{i + 1}.jpg')
    #Compacta todos os arquivos .jpg gerados em um único .zip salvo em ~/Downloads/GradCAM_Result.zip.
    zip_path = Path.home() / 'Downloads' / 'GradCAM_Result.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in download_path.iterdir():
            zipf.write(file, arcname=file.name)
    #Mensagem final informando o caminho do arquivo .zip
    print(f"\n Arquivo compactado com imagens Grad-CAM salvo em: {zip_path}")

#Mostra visualmente as primeiras imagens com Grad-CAM de uma classe específica.
#figsize=(20,5): hiperparâmetro visual — aumenta largura para mostrar várias imagens lado a lado
def show_gradcam_for_class(class_index):
    images = gradcam_refs[class_index]
    plt.figure(figsize=(20, 5))
    #Hiperparâmetro fixo = 5 imagens para visualização por classe.
    for i, raw_img in enumerate(images[:5]):
        #Mesmos passos de pré-processamento, geração e fusão do heatmap vistos na função de salvamento
        img_array = preprocess_input(img_to_array(raw_img))
        img_array = np.expand_dims(img_array, axis=0)
        heatmap = make_gradcam_heatmap(img_array, model)
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(raw_img), 0.6, heatmap_color, 0.4, 0)
        #subplot(1,5,i+1): hiperparâmetro visual — 1 linha, 5 colunas.
        #axis('off'): remove eixos para foco na imagem.
        plt.subplot(1, 5, i + 1)
        plt.imshow(overlay)
        plt.axis('off')
    #Título com o nome da classe e exibição da figura.
    plt.suptitle(f"Grad-CAM - Classe: {CLASS_NAMES[class_index]}")
    plt.show()
#Mostra Grad-CAM para todas as classes do dataset.
#NUM_CLASSES: hiperparâmetro global — número total de classes.
for i in range(NUM_CLASSES):
    show_gradcam_for_class(i)

#Pergunta ao usuário se deseja salvar os resultados; .strip().lower() garante que espaços e maiúsculas não interfiram.
#Se digitar 's' (sim), chama save_gradcam_images().
user_input = input("\nDeseja salvar as 30 primeiras imagens de cada classe com os resultados do Grad-CAM em um arquivo compactado? (s/n): ").strip().lower()
if user_input == 's':
    save_gradcam_images()

print("\n Curvas PERDA e ACURÁCIA, MATRIZ DE CONFUSÃO, ROC/AUC e MAPAS DE ATIVAÇÃO GradCAM gerados com sucesso.")
print("Obrigado, sucesso na pesquisa acadêmica científica com bons resultados!")