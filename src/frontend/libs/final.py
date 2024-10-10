# imports

import cv2 as cv
import os
import math
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
from roboflow import Roboflow
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from pytesseract import image_to_string
import re
import base64
from pathlib import Path
from collections import namedtuple
import time

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


import os
import cv2 as cv

def save_frame(video_dir, frame_num, output_dir):
    """
    Saves the extracted frame as an image file in the specified directory.
    """

    output_filename = f"frame_{frame_num}.jpg"

    if not os.path.isfile(video_dir):
        raise Exception("File not found")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    video_capture = cv.VideoCapture(video_dir)
    if not video_capture.isOpened():
        raise Exception("Could not open video")

    video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_num)

    # Check the actual frame position
    current_frame = int(video_capture.get(cv.CAP_PROP_POS_FRAMES))
    if current_frame != frame_num:
        raise Exception(f"Frame {frame_num} not found, at {current_frame} instead")

    success, frame = video_capture.read()
    if not success or frame is None:
        raise Exception(f"Could not retrieve the frame {frame_num}")

    # Optionally, check for frame integrity
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise Exception("Frame data is empty or corrupted")

    temperatures = get_frame_temepratures([frame])

    if temperatures[0][0] == 0.0 or temperatures[0][1] == 0.0:
        return save_frame(video_dir, frame_num + 1, output_dir)

    # Save the original frame without conversion (BGR format)
    output_path = os.path.join(output_dir, output_filename)
    cv.imwrite(output_path, frame)

    # Release video capture
    video_capture.release()

    return True

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

def load_and_predict(input_data, model_head, model_eyes):
  """Loads two models and returns the prediction of the last one.

  Args:
    model_head_path: Path para o modelo de cabeça.
    model_eyes_path: Path para o modelo dos olhos.
    input_data: Array de imagem contexto X.

  Returns:
    Predição do modelo de olhos.
  """
  try:
    # Predição da cabeça
    prediction_head = model_head.predict(input_data)

    # Aplicando mascara de segmentação da cabeça
    data_x = prediction_head * input_data

    # Predição dos olhos
    prediction_eye = model_eyes.predict(data_x)

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


def get_temp_olhos(image_original, image_predita, temp_tuple, percentual=80):
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
      Exibe a imagem com a temperatura desenhada no canto superior esquerdo
    """
    im = Image.fromarray(image_original.astype(np.uint8))
    draw = ImageDraw.Draw(im)

    temp_text = f"{temp:.1f}°C"
    draw.text((20, 20), temp_text, fill="red")
    im.show()

get_resized_img = lambda img: cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)

"""Pegar vídeo -> Selecionar os frames -> Realizar object detection nos frames para extrair o X -> Recortar o X -> Realizar a segmentação da cabeça e dos olhos -> Medir a temperatura de cada olho

# Object Detection
"""

def save_predictions_to_df(predictions_list):
    # Lista para armazenar todas as previsões
    all_predictions = []

    # Iterar por cada conjunto de previsões
    for prediction_set in predictions_list:
        # Extrair as previsões de cada imagem
        for prediction in prediction_set['predictions']:
            # Adicionar ao conjunto de todas as previsões
            all_predictions.append(prediction)

    # Criar o DataFrame a partir das previsões consolidadas
    df = pd.DataFrame(all_predictions)

    return df

def find_cattle(image_folder_path):
  """
  Retorna uma lista de jsons
  """
  image_files = []
  for filename in os.listdir(image_folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_files.append(os.path.join(image_folder_path, filename))

  predictions = []
  rf = Roboflow(api_key="P0Tl1nhUD7sHjcWmQCvs")
  project = rf.workspace().project("bois-teste-samuel")
  model = project.version(1).model

  for i in range(len(image_files)):
      prediction = model.predict(image_files[i], confidence=45, overlap=30).json()
      print(f"Predição {i + 1} de {len(image_files)}: {prediction}")
      predictions.append(prediction)
  return [save_predictions_to_df(predictions), predictions]

def draw_box(img, x, y, width, height, label, confidence, color=(0, 255, 0), text_color=(255, 255, 255), thickness=2, text_box_height_increase=10):
    # Calcular as coordenadas dos vértices da caixa
    x1 = int(x - width / 2)
    y1 = int(y - height / 2)
    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    # Desenhar o retângulo da caixa
    cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Texto com o rótulo e a confiança (convertido para porcentagem)
    label_text = f'{label}: {confidence * 100:.2f}%'

    # Medir o tamanho do texto para desenhar o fundo corretamente
    (text_width, text_height), baseline = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # Aumentar a altura do fundo do texto
    total_text_height = text_height + baseline + text_box_height_increase

    # Desenhar o retângulo de fundo do texto com altura aumentada
    cv.rectangle(img, (x1, y1 - total_text_height - 5), (x1 + text_width, y1), color, -1)

    # Colocar o texto sobre o fundo
    cv.putText(img, label_text, (x1, y1 - baseline - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

def plot_all_cattle_predictions(cattle_predictions):
  for i in range(len(cattle_predictions)):
    # Verifica se 'predictions' existe no dicionário e se não está vazio
    if 'predictions' in cattle_predictions[i] and cattle_predictions[i]['predictions']:
        # Obtenha o caminho da imagem a partir da primeira predição
        image_path = cattle_predictions[i]['predictions'][0].get('image_path')

        # Verifica se o caminho da imagem não é nulo ou vazio
        if image_path:
            # Ler a imagem uma vez por loop
            img = cv.imread(image_path)

            if img is not None:
                # Para cada predição da imagem, desenha a caixa
                for prediction in cattle_predictions[i].get('predictions', []):
                    x = prediction.get('x')
                    y = prediction.get('y')
                    width = prediction.get('width')
                    height = prediction.get('height')
                    label = prediction.get('class')
                    confidence = prediction.get('confidence')
                    draw_box(img, x, y, width, height, "boi", confidence)  # Desenhar a caixa na imagem

                # Exibe a imagem com todas as caixas desenhadas
                plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # Converter BGR para RGB
                plt.axis('off')  # Remover os eixos
                plt.show()  # Mostrar a imagem com todas as caixas desenhadas
            else:
                print(f"Erro ao carregar a imagem no caminho: {image_path}")
        else:
            print("Caminho da imagem está vazio.")
    else:
        print(f"Nenhuma predição encontrada para a imagem {i}")

def get_frame_temepratures(frames):
  """
  Input esperado: Uma lsita dos frames, já no formato de imagem. Não precisa estar em grayscale
  Ouput: Uma lista com tuplas (temperatura mínima:float, temperatura máxima: float)
  """

  # As imagens de referência dos números brancos, em formato de string
  all_white_encoded = [
    "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",
  ]

  # As imagens de referência dos números pretos, em formato de string
  all_black_encoded = [
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AG/sMf8AHWX+3Z4p+Pv7a3/Ep+CXwB+w/wDCM/s//wCv83+3NLuLa4/4nth/Z92uL3R4L351lznyV2IGL+f/AA7/AOCxfx2/4Ni/jt8Rf+CSXj/R/wDho3wT8PP7I/4V/d/aLXwh/Yv2+1bWb75Y7S+mufOm1VR++nfZ9mym1ZCi+f8A/BtT+wx/w8o/YT/bc/Yp/wCFo/8ACF/8Jp/wrb/ipv7E/tH7H9j1TVL/AP49/Pg8zf8AZfL/ANYu3fu527T4B/wdHf8AKdf45/8Acs/+oxpNff8A/wAHWv8AxrX/AG7PgL/wUu/Yp/4ov42+NP8AhKf+Em8a/wDIR+2fY9L0jSrf/Q7/AM+0j2WV1PD8kK7t+9suqsPr/wD4NLv2XPgT8KP+CT3hH9pfwB4G+weNviz9v/4WBrf9p3Uv9q/2XrusWtj+5klaGDyoXZP3KR785fc2DX//2Q==",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AO+/4O0fiJ/wVh8SfAnxd4A8Afs6f8I1+y54a+wf8LA+IH/CXaFef8Jl9outHksf9BkUajp32TUw0H7knz9298RACvr/AP4Ncf8AlBR8DP8AuZv/AFJ9Wo/4Ojv+UFHxz/7ln/1J9Jo/4Ncf+UFHwM/7mb/1J9Wr8gP+Do7/AIKj/t2f8N2fHP8A4Jo/8Lz/AOLJf8Uz/wAUV/wjOl/9AvSdV/4/Ps32v/j9/ff67/Y+58tfIH7Ln/Ber/grD+xd8CdC/Zo/Zo/as/4RrwT4a+1f2Jon/CC6FefZvtF1LdTfvrqxlmfdNPK/zOcbsDCgAf/Z",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APr34d/8GxvwJ+K/x2+Iv7S//BWz40/8NReNvHX9kfZNb/4Ry68E/wBlfYrVrVv3OjaksM/mwpZpyi7PsmRuaVzX5gfs8f8ADdn/AA3Z+0X/AMQk3/JEv+KR/tb/AJBf/QLk8r/kcv8AS/8Aj9/tj7v4/J5NfYH7Uf7Ufx2/4OdPjtrv/BOv/gnX45/4Rr9lzw19l/4XL8Zf7Mtbz+2vtFrFqekf8SjU4rHUbbydT0q5tP8ARpTv3ebLiIKrfr9+y5+y58Cf2LvgToX7NH7NHgb/AIRrwT4a+1f2Jon9p3V59m+0XUt1N++upZZn3TTyv8znG7AwoAH8YX7DH/BUf9uz/gmv/wAJT/wxT8c/+EL/AOE0+w/8JN/xTOl6j9s+x/aPs/8Ax/20/l7PtU/3Nu7f82dq44D9qP8Aaj+O37aPx2139pf9pfxz/wAJL428S/Zf7b1v+zLWz+0/Z7WK1h/c2sUUKbYYIk+VBnbk5Ykn/9k=",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APr34d/8GxvwJ+K/x2+Iv7S//BWz40/8NReNvHX9kfZNb/4Ry68E/wBlfYrVrVv3OjaksM/mwpZpyi7PsmRuaVzX5A/8FA/2o/jt/wAEBf8AgrF8f/2aP+CSfjn/AIVN4Jv/APhFfteif2Za695mzQra6X99rMV3MMTajePw4/12D8qIF/o+/bn/AOCXH7Cf/BSj/hFv+G1vgZ/wmn/CF/bv+EZ/4qbVNO+x/bPs/wBo/wCPC5g8zf8AZYPv7tuz5cbmz/OD/wAFA/2o/jt/wQF/4KxfH/8AZo/4JJ+Of+FTeCb/AP4RX7Xon9mWuveZs0K2ul/fazFdzDE2o3j8OP8AXYPyogU/4KB/tR/Hb/ggL/wVi+P/AOzR/wAEk/HP/CpvBN//AMIr9r0T+zLXXvM2aFbXS/vtZiu5hibUbx+HH+uwflRAvv8A/wAG9v7LnwJ/4L9fHb9pb9pf/grZ4G/4Wz42sP8AhDfsmt/2ndaD5e+11K1b9zo0tpCcw6dZpyh/1OR8zuW//9k=",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APr3/gsX/wAFi/jt4b+O1n/wST/4JJ6P/wAJL+1H4l8z7Xd/aLWz/wCEN+z2tlrK/LrNodO1H7XphvBxOPI25OZSiV8gf8GMf/N0X/ck/wDufr6A/YY/4Nqf27P+Ca//AAlP/DFP/Bbf/hC/+E0+w/8ACTf8Y2aXqP2z7H9o+z/8f+oz+Xs+1T/c27t/zZ2rj8//APg1K/Z4/bs+Pf8Awvr/AIYp/wCCiv8AwoH+yf8AhFv+Em/4tFpfir+3fN/tf7P/AMf8ifZfI8uf7mfM+0fNjy1r9v8A/gvV+1H8dv2Lv+CT3xW/aX/Zo8c/8I1428Nf2F/Ymt/2Za3n2b7Rrun2s37m6ilhfdDPKnzIcbsjDAEfyg/sMf8ABUf9uz/gmv8A8JT/AMMU/HP/AIQv/hNPsP8Awk3/ABTOl6j9s+x/aPs//H/bT+Xs+1T/AHNu7f8ANnauP//Z",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APn3/g1K/wCCXH7Cf/BSj/hfX/Da3wM/4TT/AIQv/hFv+EZ/4qbVNO+x/bP7X+0f8eFzB5m/7LB9/dt2fLjc2f1//wCIXH/ghR/0Yz/5k3xP/wDLOj/iFx/4IUf9GM/+ZN8T/wDyzr8Qf+Cgf7Ufx2/4IC/8FYvj/wDs0f8ABJPxz/wqbwTf/wDCK/a9E/sy117zNmhW10v77WYruYYm1G8fhx/rsH5UQKf8FA/2o/jt/wAEBf8AgrF8f/2aP+CSfjn/AIVN4Jv/APhFfteif2Za695mzQra6X99rMV3MMTajePw4/12D8qIF9//AODe39lz4E/8F+vjt+0t+0v/AMFbPA3/AAtnxtYf8Ib9k1v+07rQfL32upWrfudGltITmHTrNOUP+pyPmdy3/9k=",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AG/sMf8AHWX+3Z4p+Pv7a3/Ep+CXwB+w/wDCM/s//wCv83+3NLuLa4/4nth/Z92uL3R4L351lznyV2IGL/t9+y5+y58Cf2LvgToX7NH7NHgb/hGvBPhr7V/Ymif2ndXn2b7RdS3U3766llmfdNPK/wAznG7AwoAH84P/AAalf8EuP2E/+ClH/C+v+G1vgZ/wmn/CF/8ACLf8Iz/xU2qad9j+2f2v9o/48LmDzN/2WD7+7bs+XG5s8B/wUD/aj+O3/BAX/grF8f8A9mj/AIJJ+Of+FTeCb/8A4RX7Xon9mWuveZs0K2ul/fazFdzDE2o3j8OP9dg/KiBT/goH+1H8dv8AggL/AMFYvj/+zR/wST8c/wDCpvBN/wD8Ir9r0T+zLXXvM2aFbXS/vtZiu5hibUbx+HH+uwflRAvv/wDwb2/sufAn/gv18dv2lv2l/wDgrZ4G/wCFs+NrD/hDfsmt/wBp3Wg+XvtdStW/c6NLaQnMOnWacof9TkfM7lv/2Q==",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APfv2GP+DOr9hP4Cf8JT/wANreMP+F/f2t9h/wCEZ/4l+qeFf7C8r7R9o/48NWf7V5/mQffx5f2f5c+Y1fiB/wAF6v2XPgT+xd/wVi+K37NH7NHgb/hGvBPhr+wv7E0T+07q8+zfaNC0+6m/fXUssz7pp5X+ZzjdgYUAD9/v+DnL/gsX8dv+CWfwJ8F+AP2aNH+weNviz/aX9ifED7Ray/8ACM/2XdaVJN/oN1aTw3v2iG7lg+Yx+VneNzYA+gP+COv/AAR0+BP/AASA+BN54A8Aax/wkvjbxL5f/CwPiB9nurP/AISH7PdXslj/AKDJd3ENp5EN60H7kjzNu98sQF/MD/g+c/5td/7nb/3AV+/1f//Z",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APPf+De39lz4E/8ABfr47ftLftL/APBWzwN/wtnxtYf8Ib9k1v8AtO60Hy99rqVq37nRpbSE5h06zTlD/qcj5nct4B/wUD/aj+O3/BAX/grF8f8A9mj/AIJJ+Of+FTeCb/8A4RX7Xon9mWuveZs0K2ul/fazFdzDE2o3j8OP9dg/KiBff/8Ag3t/Zc+BP/Bfr47ftLftL/8ABWzwN/wtnxtYf8Ib9k1v+07rQfL32upWrfudGltITmHTrNOUP+pyPmdy3gH/AAUD/aj+O3/BAX/grF8f/wBmj/gkn45/4VN4Jv8A/hFfteif2Za695mzQra6X99rMV3MMTajePw4/wBdg/KiBT/goH+1H8dv+CAv/BWL4/8A7NH/AAST8c/8Km8E3/8Awiv2vRP7Mtde8zZoVtdL++1mK7mGJtRvH4cf67B+VEC+/wD/AAb2/sufAn/gv18dv2lv2l/+Ctngb/hbPjaw/wCEN+ya3/ad1oPl77XUrVv3OjS2kJzDp1mnKH/U5HzO5b//2Q==",
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAUAA4BAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APPf+De39lz4E/8ABfr47ftLftL/APBWzwN/wtnxtYf8Ib9k1v8AtO60Hy99rqVq37nRpbSE5h06zTlD/qcj5nct4B/wUD/aj+O3/BAX/grF8f8A9mj/AIJJ+Of+FTeCb/8A4RX7Xon9mWuveZs0K2ul/fazFdzDE2o3j8OP9dg/KiBf3+/aj/4IK/8ABJ79tH47a7+0v+0v+yn/AMJL428S/Zf7b1v/AITrXbP7T9ntYrWH9za30UKbYYIk+VBnbk5Ykn+YH/gvV+y58Cf2Lv8AgrF8Vv2aP2aPA3/CNeCfDX9hf2Jon9p3V59m+0aFp91N++upZZn3TTyv8znG7AwoAH3/AP8ABer/AIL1f8FYf2Lv+CsXxW/Zo/Zo/as/4RrwT4a/sL+xNE/4QXQrz7N9o0LT7qb99dWMsz7pp5X+ZzjdgYUAD7//AODS79lz4E/Cj/gk94R/aX8AeBvsHjb4s/b/APhYGt/2ndS/2r/Zeu6xa2P7mSVoYPKhdk/cpHvzl9zYNf/Z",
  ]

  # Transformando as listas de strings no formato de imagem novamente
  all_white = [cv.imdecode(np.frombuffer(base64.b64decode(x), np.uint8), cv.IMREAD_GRAYSCALE) for x in all_white_encoded]
  all_black = [cv.imdecode(np.frombuffer(base64.b64decode(x), np.uint8), cv.IMREAD_GRAYSCALE) for x in all_black_encoded]



  # Função que determina a cor do número da imagem
  def find_out_if_black_or_white(image_num):
    if image_num[19, 6] < 128:
      return 'black'
    else:
      return 'white'

  # Função que limpa o ruído de imagens com número branco
  def clean_white(image, threshold):
    # threshold recomendado: 190
    return np.where(image > threshold, 255, 0)

  # Função que limpa o ruído de imagens com número preto
  def clean_black(image, threshold):
    # threshold recomendado: 65
    return np.where(image < threshold, 0, 255)

  # Limpando os ruídos da conversão de string para imagem
  all_white = [clean_white(x, 190) for x in all_white]
  all_black = [clean_black(x, 65) for x in all_black]

  # Função que classifica um digito específico
  def classify_digit(digit_image):
    color = find_out_if_black_or_white(digit_image)

    if color == 'white':
      white_thresholds = [130, 150, 170, 190, 210, 230, 245]
      for threshold in white_thresholds:
        clean_digit_image = clean_white(digit_image, threshold)
        for i in range(len(all_white)):
          if np.array_equal(clean_digit_image, all_white[i]):
            return str(i)
      for i in range(len(all_white)):
        if abs(clean_white(digit_image, 190) - all_white[i]).sum() < 255*5:
          return str(i)

    elif color == 'black':
      black_thresholds = [125, 105, 85, 65, 45, 25, 15, 10]
      for threshold in black_thresholds:
        clean_digit_image = clean_black(digit_image, threshold)
        for i in range(len(all_black)):
          if np.array_equal(clean_digit_image, all_black[i]):
            return str(i)

    return '0'

  # Função que retorna as temperaturas mínimas e máximas do frame
  def receive_frame_and_extract_temperatures(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    max_tens = frame[68:88, 512:526]
    max_unit = frame[68:88, 528:542]
    max_decimal = frame[68:88, 560:574]

    min_tens = frame[404:424, 512:526]
    min_unit = frame[404:424, 528:542]
    min_decimal = frame[404:424, 560:574]

    max_number = classify_digit(max_tens) + classify_digit(max_unit) + '.' + classify_digit(max_decimal)
    min_number = classify_digit(min_tens) + classify_digit(min_unit) + '.' + classify_digit(min_decimal)

    return (float(min_number), float(max_number))

  # Corpo principal da função
  temperature_list = []
  for frame in frames:
    temperature_list.append(receive_frame_and_extract_temperatures(frame))
  return temperature_list

def save_multiple_frames(video_path, save_dir, division_factor):
  # prompt: Delete all the files inside save_dir
  # delete all the files inside save_dir
  for filename in os.listdir(save_dir):
      file_path = os.path.join(save_dir, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
              # elif os.path.isdir(file_path):
          #     shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

  frame_count = len_video_from_dir(video_path)
  for i in range(0, frame_count, division_factor):
      save_frame(video_path, i, save_dir)

def crop_all_images(X_folder_path, predictions_json):

  # Limpa o repositório
  for filename in os.listdir(X_folder_path):
      file_path = os.path.join(X_folder_path, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
              # elif os.path.isdir(file_path):
          #     shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

  for prediction in predictions_json:
    if len(prediction["predictions"]) == 0:
      continue
    cropped_images = crop_imgs_from_predicition(prediction, cv.imread(prediction["predictions"][0]["image_path"]))
    frame_path = prediction["predictions"][0]["image_path"]
    frame_name = os.path.basename(frame_path)
    frame_name_without_extension = os.path.splitext(frame_name)[0]
    for i, cropped_image in enumerate(cropped_images):
      file_name = f"{frame_name_without_extension}_X_{i}.jpg"
      output_path = os.path.join(X_folder_path, file_name)
      cv.imwrite(output_path, cv.cvtColor(cropped_image, cv.COLOR_RGB2BGR))

def segment_eyes_and_get_temperatures(frames_folder_path, X_folder_path, y_folder_path, model_head_path, model_eyes_path):

  model_head = load_model(model_head_path, compile=False) # Exemplo Path: '/content/drive/MyDrive/CattleImageRepository/G2/segmentacao_cabeca.keras'
  model_eyes = load_model(model_eyes_path, compile=False)
  images = []
  y_names = []
  
  pattern = r'frame_(\d+)_X_\d+\.jpg'
  
  formmated_frame_names = []
  
  for filename in os.listdir(X_folder_path):
    if filename.endswith(".jpg"):
      formmated_frame_names.append(filename)
      
  formmated_frame_names = sorted(formmated_frame_names, key=lambda x: int(re.search(pattern, x).group(1)))
  
  # iterate over files in X_folder_path
  for filename in formmated_frame_names:
    if filename.endswith(".jpg"):
      # load image
      img = cv.imread(os.path.join(X_folder_path, filename))
      img = get_resized_img(img)
      img = img / 255.0
      images.append(img)
      y_names.append(f"{filename[:-4]}_y.jpg")

  batched_images = np.stack(images, axis=0)
  prediction_list = load_and_predict(batched_images, model_head, model_eyes)

  eye_temperatures = []

  for i in range(len(y_names)):
    frame_name = y_names[i].split("_X_")[0]
    frame = cv.imread(os.path.join(frames_folder_path, f"{frame_name}.jpg"))
    temperature_tuple = get_frame_temepratures([frame])[0]
    print(f"{frame_name} {temperature_tuple}")
    X_name = y_names[i].replace("_y", "")
    X = cv.imread(os.path.join(X_folder_path, X_name))
    X = get_resized_img(X)
    eye_temperature = get_temp_olhos(X, prediction_list[i], temperature_tuple)
    eye_temperatures.append(eye_temperature)

  return eye_temperatures

def final_function(video_path, frames_folder_path, x_folder_path, y_folder_path, frame_interval):
  video_path = video_path
  model_head_path = '../modelo_final/models/segmentacao_cabeca.keras'
  model_eyes_path = '../modelo_final/models/modelo_olho.keras'

  # Comentar a linha abaixo quando não for a primeira vez rodando o código para esse vídeo
  save_multiple_frames(video_path, frames_folder_path, frame_interval)
  df_and_json = find_cattle(frames_folder_path)
  cattle_predictions = df_and_json[0]
  # Comentar a linha abaixo quando não for a primeira vez rodando o código para esse vídeo
  predictions_json = df_and_json[1]
  crop_all_images(x_folder_path, predictions_json)
  temperatures = segment_eyes_and_get_temperatures(frames_folder_path, x_folder_path, y_folder_path, model_head_path, model_eyes_path)
  return temperatures 

def formatted_result(video, frame_interval):
  create_video_folders(video.name)
  base_path = f"../modelo_predicao/{video.name.split('.')[0]}/"
  save_video_to_folder(base_path, video)
  video_path = base_path + video.name
  frames_folder_path = base_path + 'frames_from_video_temp_folder'
  x_folder_path = base_path + 'X_temp_folder'
  y_folder_path = base_path + 'Y_temp_folder'
  
  print(f"video_path {video_path}")
  print(f"frames_folder_path {frames_folder_path}")
  print(f"x_folder_path {x_folder_path}")
  print(f"y_folder_path {y_folder_path}")
  temperatures = final_function(video_path, frames_folder_path, x_folder_path, y_folder_path, frame_interval)
  pattern = r'frame_(\d+)_X_\d+\.jpg'
  
  formatted_frame_names = []
  
  for filename in os.listdir(x_folder_path):
    if filename.endswith(".jpg"):
      formatted_frame_names.append(filename)
      
  formatted_frame_names = sorted(formatted_frame_names, key=lambda x: int(re.search(pattern, x).group(1)))
  
  # Filtrando temperaturas e frames para que ambos tenham a mesma quantidade de elementos
  filtered_temperatures = []
  filtered_frame_names = []
    
  for temp, frame in zip(temperatures, formatted_frame_names):
      if temp is not None:
          filtered_temperatures.append(temp)
          filtered_frame_names.append(frame)
    
  return video_path, filtered_frame_names, filtered_temperatures

def save_video_to_folder(base_path, uploaded_video):
    video_path = base_path + uploaded_video.name
    
    # Abre o arquivo no modo de escrita binária
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())  # Garante que o buffer seja escrito corretamente
        f.flush()  # Limpa o buffer
        os.fsync(f.fileno())  # Garante que os dados sejam gravados no disco
    
    # Verifica se o arquivo foi salvo e o fecha antes de qualquer outra operação
    if os.path.exists(video_path):
        print(f"Vídeo salvo em: {video_path}")
    else:
        print("Erro ao salvar o vídeo.")
    
    return str(video_path)

def create_video_folders(video_name, base_path='../modelo_predicao'):
  # Extrai o nome do vídeo sem extensão para criar as pastas
  video_stem = Path(video_name).stem
  
  # Define o caminho da pasta principal, com o nome do vídeo
  main_folder = Path(base_path) / video_stem
  main_folder.mkdir(parents=True, exist_ok=True)
  
  # Define as subpastas: frames, X e Y
  frames_folder = main_folder / 'frames_from_video_temp_folder'
  x_folder = main_folder / 'X_temp_folder'
  y_folder = main_folder / 'Y_temp_folder'
  
  # Cria as subpastas
  frames_folder.mkdir(parents=True, exist_ok=True)
  x_folder.mkdir(parents=True, exist_ok=True)
  y_folder.mkdir(parents=True, exist_ok=True)
