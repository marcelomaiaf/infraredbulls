import os

import pandas as pd
import streamlit as st
from PIL import Image

logo  = "./imgs/Infrar-removebg-preview.png"
VIDEO_URL = "https://www.youtube.com/watch?v=kbLSO3eJKvE"

col1, col2 = st.columns(2)

with col1:
    st.title("InfraRedBulls")

with col2:
    st.image(logo)

st.write("Nosso projeto desenvolve uma solução inovadora para monitorar a saúde do gado em tempo real, utilizando a inteligência artificial e termografia infravermelha. Ao focar na região ocular, nossa tecnologia mede a temperatura corporal de forma precisa, identificando sinais de estresse térmico e doenças de maneira precoce. Com o uso de redes neurais convolucionais (CNNs), automatizamos a segmentação das imagens térmicas, permitindo uma análise contínua e eficiente, mesmo em ambientes com alta rotatividade de animais. Essa abordagem revolucionária promete transformar o monitoramento de saúde bovina, garantindo bem-estar animal e otimização de processos.")
st.video(VIDEO_URL)
