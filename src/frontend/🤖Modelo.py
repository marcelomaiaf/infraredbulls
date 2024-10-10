import os
import tempfile
import zipfile
from io import BytesIO
from libs.outputter import create_df, append_row
from libs.final import formatted_result
import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import re


logo  = "imgs/Infrar-removebg-preview.png"

st.logo(logo, icon_image=logo, size="large")

def len_video_from_dir(file_path):
    """
    Parameters:
      file_path: Source video file path
    Raises:
      Exception: File not found
      Exception: Video corrupted or of invalid format
    Returns: len of video
    """
    if not os.path.isfile(file_path):
        raise Exception("File not found")

    video_capture = cv.VideoCapture(file_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video")

    return int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))


def select_frame(file_path, frame_num):
    """
    Parameters:
      file_path: Source video file path
      frame_num: Frame to fetch in video
    Raises:
      Exception: File not found
      Exception: Video corrupted or of invalid format
      Exception: Frame not in video
    Returns: np.array representing video frame
    """
    if not os.path.isfile(file_path):
        raise Exception("File not found")

    video_capture = cv.VideoCapture(file_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video")

    video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_num)

    success, frame = video_capture.read()

    if not success:
        raise Exception(
            f"Could not get desired frame. Make sure frame is in scope (0, {int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))})"
        )

    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)


def mock_model_inference():
    """Mock de um DataFrame gerado pelo modelo

    Returns:
        pd.DataFrame: DataFrame com os resultados da inferência do modelo.
    """
    data = [
        {"frame": "1", "boi_do_frame": 1, "qtd_bois": 2, "temperatura": "36", "com_febre": "Não", "image_path": "./uploads/frame1.png"},
        {"frame": "1", "boi_do_frame": 2, "qtd_bois": 2, "temperatura": "39", "com_febre": "Sim", "image_path": "./uploads/frame1.png"},
        {"frame": "1", "boi_do_frame": 101, "qtd_bois": 2, "temperatura": "36", "com_febre": "Não", "image_path": "./uploads/frame1.png"},
        {"frame": "2", "boi_do_frame": 102, "qtd_bois": 1, "temperatura": "36", "com_febre": "Sim", "image_path": "./uploads/frame1.png"},
        {"frame": "3", "boi_do_frame": 103, "qtd_bois": 1, "temperatura": "36", "com_febre": "Não", "image_path": "./uploads/frame1.png"}
    ]
    return pd.DataFrame(data)

def display_image(image_name, video_name):
    """Exibe uma imagem no frontend do Streamlit.

    Args:
        image_path (str): Caminho para a imagem a ser exibida.
    """
    
    pattern = r"(frame_\d+)_\w+"
    print(f"img name: {image_name}")
    print(f"regex = {re.sub(pattern, r"\1", image_name)}")
    path = f"../modelo_predicao/{video_name.split('.')[0]}/frames_from_video_temp_folder/{re.sub(pattern, r"\1", image_name)}"
    print(f"path: {path}")
    try:
        img = Image.open(path)
        st.image(img, caption=f"Imagem: {os.path.basename(image_name)} from {video_name}", use_column_width=True)
    except FileNotFoundError:
        st.error(f"Imagem não encontrada: {image_name}")

def save_all_frames(file_path):
    """
    Parameters:
      file_path: Path to the video file
    Returns:
      BytesIO: A zip file containing all frames from the video
    """
    video_capture = cv.VideoCapture(file_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video")
    
    total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for frame_num in range(total_frames):
            success, frame = video_capture.read()
            if not success:
                break
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            frame_name = f"frame_{frame_num:04d}.png"
            
            _, img_encoded = cv.imencode(".png", cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))
            zip_file.writestr(frame_name, img_encoded.tobytes())
    
   
    zip_buffer.seek(0)#ponteiro começando no 0
    
    return zip_buffer

filename = None
# Título da aplicação
st.title("InfraRedBulls - Detecção de Bois")

with st.container():
    uploaded_video = st.file_uploader("Selecione o vídeo", type=["mp4", "mov", "avi", "mkv"])

    if st.button("Realizar predição", key=None,type="primary", icon="🐮", use_container_width=True):
        if uploaded_video is not None:
            video_name = uploaded_video.name
            print(video_name)
            st.video(uploaded_video)
            
            # Salvar o vídeo em um arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_video.read())
                temp_file_path = temp_file.name

            # Carregar o modelo e realizar a inferência
            video_path, frame_names, temperatures = formatted_result(uploaded_video, 100)

            print(len(frame_names), len(temperatures))

            df = create_df()

            for i in range(len(frame_names)):
                df = append_row(df, video_name, frame_names[i], temperatures[i])

            # Tabs para exibir os resultados do modelo e os bois com febre
            resumo_modelo, bois_com_febre = st.tabs(["Resultado do modelo", "Bois com Febre"])

            # Exibindo os resultados do modelo - todo resultado do modelo será exibido aqui
            with resumo_modelo:
                st.subheader("Resultado do Modelo")
                st.write("O modelo detectou os seguintes bois:")
                
                # Campo de texto para filtrar pelo nome do frame
                frame_filter = st.text_input("Filtrar pelo Nome do Frame", "")
                
                # Filtrando DataFrame com base no input do usuário
                if frame_filter:
                    filtered_df = df[df['frame_id'].str.contains(frame_filter, case=False)]
                else:
                    filtered_df = df

                # Exibindo DataFrame filtrado
                st.dataframe(filtered_df, width=700)
                
                # Exibindo detalhes dos bois
                for index, row in filtered_df.iterrows():
                    with st.expander(f"Detalhes do Boi {row['boi_id']} do frame {row['frame_id']}"):
                        st.write(f"**Frame:** {row['frame_id']}")
                        st.write(f"**Boi do Frame:** {row['boi_id']}")
                        st.write(f"**Temperatura:** {row['temperatura °C']} °C")
                        st.write(f"**Com Febre:** {'Sim' if row['febre'] else 'Não'}")
                        display_image(row['frame_id'], video_name)

            # Exibindo os bois com febre - somente bois com febre serão exibidos aqui
            with bois_com_febre:
                st.subheader("Bois com Febre")
                st.write("Os seguintes bois foram detectados com febre:")
                
                # Filtrando DataFrame para exibir apenas os bois com febre
                bois_com_febre = df[df["febre"] == True]
                bois_com_febre = bois_com_febre.reset_index(drop=True)
                st.dataframe(bois_com_febre, width=700)

                # Exibindo detalhes dos bois
                for index, row in bois_com_febre.iterrows():
                    with st.expander(f"Detalhes do Boi {row['boi_id']} do frame {row['frame_id']}"):
                        st.write(f"**Frame:** {row['frame_id']}")
                        st.write(f"**Boi do Frame:** {row['boi_id']}")
                        st.write(f"**Temperatura:** {row['temperatura °C']} °C")
                        st.write(f"**Com Febre:** {'Sim' if row['febre'] else 'Não'}")
                        display_image(row['frame_id'], video_name)

            # Gerar o arquivo ZIP contendo todos os frames
            zip_file = save_all_frames(video_path)
            st.download_button(
                    label="Download dos frames",
                    data=zip_file,
                    file_name=f"frames_{uploaded_video.name}.zip",
                    mime="application/zip",
                    icon=":material/download:", 
                    use_container_width=True,
                    type="primary"
                )
        else:
            st.error('Faça o upload do vídeo para realizar a predição', icon="🚨")
    