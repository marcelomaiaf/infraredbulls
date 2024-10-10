import cv2 as cv
import os
import math
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def len_video_from_dir(dir):
    """
    Parameters:
      dir: Source video directory
    Raises:
      Exception: File not found
      Exception: Video corrupted or of invalid format
    Returns: len of video
    """
    if not os.path.isfile(dir):
        raise Exception("File not found")

    video_capture = cv.VideoCapture(dir)
    if not video_capture.isOpened():
        raise Exception("Could not open video")

    return int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))


def select_frame(dir, frame_num):
    """
    Parameters:
      dir: Source video directory
      frame_num: Frame to fetch in video
    Raises:
      Exception: File not found
      Exception: Video corrupted or of invalid format
      Exception: Frame not in video
    Returns: np.array representing video frame
    """
    if not os.path.isfile(dir):
        raise Exception("File not found")

    video_capture = cv.VideoCapture(dir)
    if not video_capture.isOpened():
        raise Exception("Could not open video")

    video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_num)

    success, frame = video_capture.read()

    if not success:
        raise Exception(
            f"Could not get desired frame. Make sure frame, Make sure frame is in scope (0, {int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))})"
        )

    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)


get_frames = lambda dir: [
    (i, select_frame(dir, i)) for i in range(0, len_video_from_dir(dir), 20)
]


def crop_imgs_from_predicition(pred, img):
    """
    Parameters:
      pred: Roboflow prediction json
      img: Image to be cropped
    Returns: List of cropped images from predictions or None if no images were cropped
    """
    pred = pred["predictions"]
    imgs = []
    for p in pred:
        h = int(math.floor(p["height"]))
        w = int(math.floor(p["width"]))
        py = (int(p["y"]) - (h // 2), int(p["y"]) + (h // 2))
        px = (int(p["x"]) - (w // 2), int(p["x"]) + (w // 2))
        imgs.append(img[py[0]:py[1], px[0]:px[1]])
    return imgs if len(imgs) != 0 else None

def load_and_predict(input_data, model_head_path, model_eyes_path):
  """Loads two models and returns the prediction of the last one.

  Args:
    model_head_path: Path para o modelo de cabeça.
    model_eyes_path: Path para o modelo dos olhos.
    input_data: Array de imagem contexto X.

  Returns:
    Predição do modelo de olhos.
  """
  try:
    # Carrega o modelo de cabeça
    model1 = load_model(model_head_path, compile=False) # Exemplo Path: '/content/drive/MyDrive/CattleImageRepository/G2/segmentacao_cabeca.keras'

    # Carrega o modelo de olhos
    model2 = load_model(model_eyes_path, compile=False) # Exemplo Path: '/content/drive/MyDrive/CattleImageRepository/G2/modelo_olho.keras'

    # Predição da cabeça
    prediction_head = model1.predict(input_data)

    # Aplicando mascara de segmentação da cabeça
    data_x = prediction_head * input_data

    # Predição dos olhos
    prediction_eye = model2.predict(data_x)

    # Retorna segmentação dos olhos
    return prediction_eye
  except Exception as e:
    print(f"Erro carregando os modelos: {e}")
    raise
    return None


def only_eyes(image_original,image_predita):
  """
    Parameters:
      image_original: Contexto  X image 128x128
      image_predita: output do modelo de sementação dos olhos image 128x128
    Returns: Imagem original com área preta fora dos olhos
    """
  img1 = image_predita
  img2 = image_original
  mask = img1 > 0.5 #seleciono apenas os pixeis que não são pretos na imagem predita
  img2_eyes = np.zeros_like(img2) #crio uma imagem vazia com as mesmas dimensões da imagem original

  img2_eyes[mask] = img2[mask]# faço a máscara dos olhos na imagem original

  return img2_eyes

def get_temp_from_pixel(pixel, temp_tuple):
    """
    Parameters:
      pix: rgb color tuple representing image pixel
      temp_tuple: (min, max) temperature tuple representing the temperature scale
    Returns: °C tempertature from pixel
    """
    MIN_TEMP_PIXEL = 32 / 255.0
    MAX_TEMP_PIXEL = 159 / 255.0
    TEMP_PIXEL_RANGE = MAX_TEMP_PIXEL - MIN_TEMP_PIXEL

    if pixel < MIN_TEMP_PIXEL: pixel = MIN_TEMP_PIXEL
    if pixel > MAX_TEMP_PIXEL: pixel = MAX_TEMP_PIXEL

    min_temp = min(temp_tuple)
    max_temp = max(temp_tuple)
    temp_range = max_temp - min_temp

    temp = min_temp + ((pixel - MIN_TEMP_PIXEL) / TEMP_PIXEL_RANGE) * temp_range

    return temp


def get_temp_olhos(image_original, image_predita, temp_tuple, percentual=100):
    """
    Parameters:
      image_original: Imagem de contexto 128x128
      image_predita: Output do modelo de segmentação dos olhos (imagem 128x128)
      temp_tuple: (min, max) tupla de temperatura representando o intervalo de temperatura
      percentual: Percentual dos pixels mais intensos a serem considerados para cálculo da temperatura
    Returns: Temperatura em °C dos olhos bovinos
    """
    img = only_eyes(image_original, image_predita)
    img_nonzero = img[img > 0.1]  # Considera apenas os pixels não nulos

    if len(img_nonzero) == 0:
        return None  # Se não houver pixels válidos, retorna None

    # Ordena os pixels e seleciona o percentual mais quente
    threshold = np.percentile(img_nonzero, percentual)
    selected_pixels = img_nonzero[img_nonzero >= threshold]

    # Calcula a média dos pixels selecionados
    media = np.mean(selected_pixels)

    return get_temp_from_pixel(media, temp_tuple)

def draw_temp(image_original, temp):
    """
    Parameters:
      image_original: Imagem original no formato np.array
      temp: Temperatura que será desenhada na imagem
    Returns:
      Imagem com a temperatura desenhada no canto superior esquerdo
    """
    im = Image.fromarray(image_original.astype(np.uint8))
    draw = ImageDraw.Draw(im)
    temp_text = f" A temperatura é {temp:.1f}°C"

    draw.text((20, 20), temp_text, fill="red")

    img_numpy = np.array(im)

    plt.imshow(img_numpy)
    plt.axis('off')  
    plt.show()

get_resized_img = lambda img: cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
