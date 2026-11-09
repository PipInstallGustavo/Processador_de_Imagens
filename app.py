import streamlit as st

if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun

from PIL import Image
import numpy as np
import io

from modulos import intensidade, espacial, cores
from paginas import fourier as pagina_fourier

st.set_page_config(page_title="Processador de Imagens", layout="wide")
st.title("Processador de Imagens")
st.sidebar.title("Menu Principal")

categoria = st.sidebar.selectbox(
    "Escolha a categoria de processamento:",
    [
        "Processamento de Intensidade",
        "Filtragem e Operações Espaciais",
        "Transformada de Fourier",
        "Processamento de Cores",
    ],
)

# ──────────────────────────────────────────────────────────────
# Fourier tem upload próprio — renderiza completamente separado
# ──────────────────────────────────────────────────────────────
if categoria == "Transformada de Fourier":
    pagina_fourier.render()

# ──────────────────────────────────────────────────────────────
# Demais categorias: upload único na sidebar
# ──────────────────────────────────────────────────────────────
else:
    uploaded_file = st.sidebar.file_uploader(
        "Envie uma imagem para processar",
        type=["jpg", "jpeg", "png", "bmp", "tif"],
    )

    if uploaded_file is not None:
        img_original = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img_original, use_container_width=True)

        img_result = None

        # ── Processamento de Intensidade ──────────────────────
        if categoria == "Processamento de Intensidade":
            operacao = st.sidebar.selectbox(
                "Operação:",
                [
                    "Limiarização",
                    "Negativo",
                    "Brilho",
                    "Correção de Gama",
                    "Linear por Partes (Configurável)",
                    "Alargamento de Contraste",
                    "Esteganografia (Esconder)",
                    "Esteganografia (Revelar)",
                    "Plotar Histograma",
                    "Equalização de Histograma",
                ],
            )

            if operacao == "Limiarização":
                limiar = st.sidebar.slider("Limiar", 0.0, 1.0, 0.5)
                img_result = intensidade.limiarizacao(img_original, limiar)
            elif operacao == "Negativo":
                img_result = intensidade.negativo(img_original)
            elif operacao == "Brilho":
                valor = st.sidebar.slider("Valor", -255, 255, 50)
                img_result = intensidade.brilho(img_original, valor)
            elif operacao == "Correção de Gama":
                c = st.sidebar.number_input("C", value=1.0)
                gama = st.sidebar.slider("Gama", 0.1, 5.0, 1.0)
                img_result = intensidade.correcao_gama(img_original, c, gama)
            elif operacao == "Linear por Partes (Configurável)":
                st.sidebar.write("Defina os breakpoints da transformação linear por partes:")
                st.sidebar.caption("Cada linha: `r_entrada,s_saida` (0-255). Ex: 0,0 / 100,50 / 200,230 / 255,255")
                pontos_str = st.sidebar.text_area("Pontos (r,s por linha)", "0,0\n80,20\n180,240\n255,255", key="lpp_pontos")
                try:
                    pontos = []
                    for linha in pontos_str.strip().split("\n"):
                        r_in, s_out = map(float, linha.split(","))
                        pontos.append((int(r_in), int(s_out)))
                    pontos.sort(key=lambda p: p[0])
                    img_result = intensidade.linear_por_partes(img_original, pontos)
                except Exception as e:
                    st.sidebar.error(f"Erro: {e}")
            elif operacao == "Alargamento de Contraste":
                rmin = st.sidebar.slider("R_min", 0, 255, 50)
                rmax = st.sidebar.slider("R_max", 0, 255, 200)
                if rmin < rmax:
                    img_result = intensidade.contrast_stretching(
                        img_original, rmin, rmax
                    )
                else:
                    st.error("R_min deve ser menor que R_max")
            elif operacao == "Esteganografia (Esconder)":
                msg = st.sidebar.text_input("Mensagem para esconder:")
                if st.sidebar.button("Esconder") and msg:
                    img_result = intensidade.esteganografia_esconder(
                        img_original, msg
                    )
            elif operacao == "Esteganografia (Revelar)":
                if st.sidebar.button("Revelar"):
                    msg = intensidade.esteganografia_revelar(img_original)
                    st.success(f"Mensagem oculta: {msg}")
                    img_result = img_original
            elif operacao == "Plotar Histograma":
                img_result = intensidade.plotar_histograma(img_original)
            elif operacao == "Equalização de Histograma":
                img_result = intensidade.equalizacao_histograma(img_original)

        # ── Filtragem Espacial ────────────────────────────────
        elif categoria == "Filtragem e Operações Espaciais":
            operacao = st.sidebar.selectbox(
                "Operação:",
                [
                    "Filtro Genérico (Convolução)",
                    "Média",
                    "Média Ponderada",
                    "Gaussiano",
                    "Mediana",
                    "Laplaciano",
                    "Aguçamento Laplaciano",
                    "High-Boost",
                    "Sobel X",
                    "Sobel Y",
                    "Gradiente (Magnitude)",
                    "Fuzzy",
                    "Escala (Vizinho Mais Próximo)",
                    "Escala (Bilinear)",
                    "Rotação (Vizinho Mais Próximo)",
                    "Rotação (Bilinear)",
                    "Demonstração de Aliasing (Moiré)",
                ],
            )

            normalizar = st.sidebar.checkbox(
                "Normalizar para exibição (tons de cinza)", value=False
            )

            if operacao == "Filtro Genérico (Convolução)":
                st.sidebar.write(
                    "Defina a matriz (separada por espaços e novas linhas):"
                )
                matriz_str = st.sidebar.text_area(
                    "Matriz", "0 -1 0\n-1 5 -1\n0 -1 0"
                )
                if st.sidebar.button("Aplicar"):
                    try:
                        linhas = matriz_str.strip().split("\n")
                        matriz = np.array(
                            [[float(v) for v in l.split()] for l in linhas]
                        )
                        img_result = espacial.aplicar_filtro_generico(
                            img_original, matriz, normalizar
                        )
                    except Exception as e:
                        st.error(f"Erro na matriz: {e}")
            elif operacao == "Média":
                tamanho = st.sidebar.slider("Tamanho do kernel", 3, 15, 3, step=2)
                img_result = espacial.filtro_media(img_original, tamanho)
            elif operacao == "Média Ponderada":
                tamanho = st.sidebar.slider("Tamanho do kernel", 3, 15, 3, step=2)
                img_result = espacial.filtro_media_ponderada(img_original, tamanho)
            elif operacao == "Gaussiano":
                tamanho = st.sidebar.slider("Tamanho do kernel", 3, 15, 3, step=2)
                sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0)
                img_result = espacial.filtro_gaussiano(img_original, tamanho, sigma)
            elif operacao == "Mediana":
                tamanho = st.sidebar.slider("Tamanho da janela", 3, 15, 3, step=2)
                img_result = espacial.filtro_mediana(img_original, tamanho)
            elif operacao == "Laplaciano":
                img_result = espacial.filtro_laplaciano(img_original, normalizar)
            elif operacao == "Aguçamento Laplaciano":
                img_result = espacial.aguacamento_laplaciano(img_original)
            elif operacao == "High-Boost":
                k = st.sidebar.slider("Fator k", 1.0, 5.0, 1.5)
                tamanho = st.sidebar.slider(
                    "Tamanho do kernel gaussiano", 3, 15, 3, step=2
                )
                img_result = espacial.filtro_high_boost(img_original, k, tamanho)
            elif operacao == "Sobel X":
                img_result = espacial.filtro_sobel_x(img_original, normalizar)
            elif operacao == "Sobel Y":
                img_result = espacial.filtro_sobel_y(img_original, normalizar)
            elif operacao == "Gradiente (Magnitude)":
                img_result = espacial.gradiente_magnitude(img_original, normalizar)
            elif operacao == "Fuzzy":
                img_result = espacial.intensificacao_fuzzy(img_original)
            elif operacao == "Escala (Vizinho Mais Próximo)":
                sx = st.sidebar.slider("Fator X", 0.1, 5.0, 2.0)
                sy = st.sidebar.slider("Fator Y", 0.1, 5.0, 2.0)
                img_result = espacial.escala_vizinho(img_original, sx, sy)
            elif operacao == "Escala (Bilinear)":
                sx = st.sidebar.slider("Fator X", 0.1, 5.0, 2.0)
                sy = st.sidebar.slider("Fator Y", 0.1, 5.0, 2.0)
                img_result = espacial.escala_bilinear(img_original, sx, sy)
            elif operacao == "Rotação (Vizinho Mais Próximo)":
                angulo = st.sidebar.slider("Ângulo", -180, 180, 45)
                pivo_x = st.sidebar.number_input(
                    "Pivô X (0 = centro)", value=0
                )
                pivo_y = st.sidebar.number_input(
                    "Pivô Y (0 = centro)", value=0
                )
                px = pivo_x if pivo_x != 0 else None
                py = pivo_y if pivo_y != 0 else None
                img_result = espacial.rotacao_vizinho(
                    img_original, angulo, px, py
                )
            elif operacao == "Demonstração de Aliasing (Moiré)":
                fator = st.sidebar.slider("Fator de Subamostragem", 2, 16, 4)
                st.sidebar.info(
                    "Subamostragem por fator N e reconstrução por vizinho mais próximo. "
                    "Quanto maior o fator, mais visível o efeito Moiré/Aliasing."
                )
                img_result = espacial.demonstrar_aliasing(img_original, fator)
            elif operacao == "Rotação (Bilinear)":
                angulo = st.sidebar.slider("Ângulo", -180, 180, 45)
                pivo_x = st.sidebar.number_input(
                    "Pivô X (0 = centro)", value=0
                )
                pivo_y = st.sidebar.number_input(
                    "Pivô Y (0 = centro)", value=0
                )
                px = pivo_x if pivo_x != 0 else None
                py = pivo_y if pivo_y != 0 else None
                img_result = espacial.rotacao_bilinear(
                    img_original, angulo, px, py
                )

        # ── Processamento de Cores ────────────────────────────
        elif categoria == "Processamento de Cores":
            operacao = st.sidebar.selectbox(
                "Operação:",
                [
                    "Tons de Cinza (Média)",
                    "Tons de Cinza (Ponderada NTSC)",
                    "Negativo (Colorido)",
                    "Visualizar Canais HSI (RGB→HSI)",
                    "Chroma-Key",
                    "Equalização (HSI)",
                    "Ajuste Matiz, Sat, Brilho",
                    "Ajuste de Canais RGB",
                    "Sépia",
                ],
            )

            if operacao == "Tons de Cinza (Média)":
                img_result = cores.escala_cinza_media(img_original)
            elif operacao == "Tons de Cinza (Ponderada NTSC)":
                img_result = cores.escala_cinza_ponderada(img_original)
            elif operacao == "Negativo (Colorido)":
                img_result = intensidade.negativo(img_original)
            elif operacao == "Visualizar Canais HSI (RGB→HSI)":
                hsi_img = cores.visualizar_hsi(img_original)
                with col2:
                    st.subheader("Canais H, S, I separados")
                    st.image(hsi_img, use_container_width=True)
                img_result = None
            elif operacao == "Chroma-Key":
                bg_file = st.sidebar.file_uploader(
                    "Envie a imagem de Fundo", type=["jpg", "png", "jpeg"]
                )
                tolerancia = st.sidebar.slider(
                    "Tolerância do verde",
                    min_value=0,
                    max_value=60,
                    value=20,
                    help="Controla a faixa de tons verdes substituídos pelo fundo.",
                )
                if bg_file:
                    bg_img = Image.open(bg_file)
                    img_result = cores.chroma_key(
                        img_original, bg_img, tolerancia=tolerancia
                    )
            elif operacao == "Equalização (HSI)":
                img_result = cores.equalizar_hsi(img_original)
            elif operacao == "Ajuste Matiz, Sat, Brilho":
                h_shift = st.sidebar.slider("Deslocamento Matiz (H)", -180, 180, 0)
                s_scale = st.sidebar.slider("Escala Saturação (S)", 0.0, 3.0, 1.0)
                i_shift = st.sidebar.slider("Deslocamento Brilho (I)", -100, 100, 0)
                img_result = cores.ajustar_hsi(
                    img_original, h_shift, s_scale, i_shift
                )
            elif operacao == "Ajuste de Canais RGB":
                r = st.sidebar.slider("Canal Vermelho (R)", 0.0, 3.0, 1.0)
                g = st.sidebar.slider("Canal Verde (G)", 0.0, 3.0, 1.0)
                b = st.sidebar.slider("Canal Azul (B)", 0.0, 3.0, 1.0)
                img_result = cores.ajustar_canais(img_original, r, g, b)
            elif operacao == "Sépia":
                img_result = cores.sepia(img_original)

        # ── Exibe resultado e botão de download ───────────────
        if img_result is not None:
            with col2:
                st.subheader("Resultado")
                st.image(img_result, use_container_width=True)

            buf = io.BytesIO()
            img_result.save(buf, format="PNG")
            st.download_button(
                label="Baixar Imagem Processada",
                data=buf.getvalue(),
                file_name="resultado.png",
                mime="image/png",
            )
    else:
        st.info("Faça o upload de uma imagem na barra lateral para começar.")
