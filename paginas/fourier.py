"""
paginas/fourier.py
==================
Página de Transformada de Fourier integrada ao app principal.

Suporta imagens em escala de cinza e RGB, com:
  - DFT manual (O(N²) por dimensão);
  - FFT manual Cooley-Tukey (O(N log N));
  - FFT do NumPy como alternativa rápida;
  - visualização logarítmica e centralizada do espectro;
  - edição interativa do espectro com pincéis preto, branco e cinza;
  - transformada inversa e download do resultado.

O backend modulos/frequencia.py realiza o processamento RGB canal a canal.
"""

import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image

from modulos import frequencia as freq

# Compatibilidade com algumas versões do Streamlit / drawable-canvas.
import streamlit.elements.image as _st_image
try:
    import streamlit.elements.lib.image_utils as _image_utils
    if not hasattr(_st_image, "image_to_url"):
        _st_image.image_to_url = _image_utils.image_to_url
except ImportError:
    pass

from streamlit_drawable_canvas import st_canvas


# Chaves utilizadas nesta página. Mantê-las agrupadas facilita limpar
# resultados antigos quando o usuário troca a imagem ou o método.
CHAVES_FOURIER = [
    "fourier_imagem_proc",
    "fourier_espectro_original",
    "fourier_espectro_editado",
    "fourier_h_orig",
    "fourier_w_orig",
    "fourier_metodo_calc",
    "fourier_resultado",
    "fourier_assinatura",
]


def _carregar_imagem(uploaded_file):
    """Preserva imagens em tons de cinza e converte as demais para RGB."""
    imagem = Image.open(io.BytesIO(uploaded_file.getvalue()))

    # Modos que representam imagens monocromáticas.
    if imagem.mode in ("1", "L", "I", "I;16", "F"):
        imagem = imagem.convert("L")
        tipo = "Escala de cinza"
    else:
        # Inclui RGB, RGBA, CMYK e imagens com paleta de cores.
        imagem = imagem.convert("RGB")
        tipo = "RGB (colorida)"

    return imagem, tipo


def _limpar_estado_fourier():
    for chave in CHAVES_FOURIER:
        st.session_state.pop(chave, None)


def render():
    """Renderiza a página completa de Fourier dentro do app."""

    st.header("Transformada de Fourier de Imagens")
    st.write(
        "Carregue uma imagem em escala de cinza ou colorida, escolha o método, "
        "edite o espectro e reconstrua a imagem por meio da transformada inversa."
    )

    # ------------------------------------------------------------
    # 1. Entrada e configurações
    # ------------------------------------------------------------
    st.sidebar.header("Configurações de Fourier")

    arquivo = st.sidebar.file_uploader(
        "Imagem para Fourier",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="fourier_upload",
    )

    metodo = st.sidebar.selectbox(
        "Método de transformação",
        ["FFT manual", "DFT manual", "FFT NumPy"],
        key="fourier_metodo",
    )

    if metodo == "DFT manual":
        st.sidebar.warning(
            "A DFT manual possui alto custo computacional. Para evitar demora, "
            "a imagem é reduzida quando sua maior dimensão ultrapassa 64 pixels."
        )
    elif metodo == "FFT manual":
        st.sidebar.info(
            "A FFT manual utiliza Cooley-Tukey. Dimensões que não são "
            "potências de 2 recebem preenchimento com zeros (zero-padding)."
        )
    else:
        st.sidebar.info(
            "A FFT do NumPy é disponibilizada como alternativa rápida para imagens maiores."
        )

    if arquivo is None:
        _limpar_estado_fourier()
        st.info("📂 Faça o upload de uma imagem na barra lateral para começar.")
        return

    # Lê a imagem sem forçar conversão para cinza: imagens monocromáticas
    # continuam 2D e imagens coloridas são processadas como RGB.
    imagem_pil, tipo_imagem = _carregar_imagem(arquivo)
    imagem_original = np.asarray(imagem_pil, dtype=float)

    # A assinatura impede que um resultado antigo continue aparecendo quando
    # o usuário troca o arquivo ou o método de transformação.
    assinatura = hashlib.sha1(
        arquivo.getvalue() + metodo.encode("utf-8")
    ).hexdigest()
    if st.session_state.get("fourier_assinatura") != assinatura:
        _limpar_estado_fourier()
        st.session_state["fourier_assinatura"] = assinatura

    imagem_proc = freq.preparar_imagem(imagem_original, metodo)
    altura_proc, largura_proc = imagem_proc.shape[:2]

    st.caption(
        f"Formato detectado: **{tipo_imagem}** · "
        f"Dimensões de entrada: **{imagem_original.shape[1]} × "
        f"{imagem_original.shape[0]} px** · "
        f"Dimensões processadas: **{largura_proc} × {altura_proc} px**"
    )

    # ------------------------------------------------------------
    # 2. Cálculo da transformada
    # ------------------------------------------------------------
    if st.button(
        "▶ Calcular Transformada",
        type="primary",
        key="btn_calcular_fourier",
    ):
        with st.spinner("Calculando transformada... Aguarde."):
            espectro, h_orig, w_orig = freq.calcular_transformada(
                imagem_proc, metodo
            )

        st.session_state["fourier_imagem_proc"] = imagem_proc
        st.session_state["fourier_espectro_original"] = espectro
        st.session_state["fourier_espectro_editado"] = espectro.copy()
        st.session_state["fourier_h_orig"] = h_orig
        st.session_state["fourier_w_orig"] = w_orig
        st.session_state["fourier_metodo_calc"] = metodo
        st.session_state.pop("fourier_resultado", None)
        st.success("Transformada calculada com sucesso!")

    # Só mostra o editor depois que uma transformada foi calculada.
    if "fourier_espectro_original" not in st.session_state:
        return

    imagem_proc = st.session_state["fourier_imagem_proc"]
    espectro_original = st.session_state["fourier_espectro_original"]
    espectro_editado = st.session_state["fourier_espectro_editado"]
    h_orig = st.session_state["fourier_h_orig"]
    w_orig = st.session_state["fourier_w_orig"]
    metodo_atual = st.session_state["fourier_metodo_calc"]

    # O backend retorna uma matriz 2D para cinza ou uma imagem RGB para cor.
    imagem_espectro_vis = freq.visualizar_espectro(espectro_editado)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagem de entrada")
        st.image(
            freq.imagem_para_exibicao(imagem_proc),
            clamp=True,
            use_container_width=True,
        )
    with col2:
        st.subheader("Espectro de Fourier")
        st.image(
            imagem_espectro_vis,
            clamp=True,
            use_container_width=True,
        )
        if imagem_proc.ndim == 3:
            st.caption(
                "Espectro RGB: cada canal é transformado e normalizado "
                "individualmente para visualização."
            )
        else:
            st.caption("Magnitude em escala logarítmica, com frequência zero centralizada.")

    st.divider()

    # ------------------------------------------------------------
    # 3. Edição do espectro
    # ------------------------------------------------------------
    st.subheader("✏️ Editar Espectro")
    st.write(
        "Desenhe sobre o espectro para modificar as frequências. "
        "**Preto** zera a magnitude, **branco** maximiza a magnitude e "
        "**cinza suave** define um valor intermediário."
    )

    if imagem_proc.ndim == 3:
        st.info(
            "Em imagens RGB, a mesma região desenhada é aplicada aos espectros "
            "dos canais vermelho, verde e azul. A fase de cada canal é preservada."
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tipo_pincel = st.selectbox(
            "Pincel",
            ["Preto", "Branco", "Cinza suave"],
            key="fourier_pincel",
        )
    with col_b:
        tamanho_pincel = st.slider(
            "Tamanho do pincel", 2, 40, 10, key="fourier_tamanho"
        )
    with col_c:
        intensidade = st.slider(
            "Intensidade do cinza",
            0.0,
            1.0,
            0.5,
            step=0.05,
            key="fourier_intensidade",
        )

    # Mantém a proporção do espectro ao limitar o tamanho do canvas.
    altura_espectro, largura_espectro = imagem_espectro_vis.shape[:2]
    escala_canvas = min(
        1.0,
        512 / altura_espectro,
        768 / largura_espectro,
    )
    altura_canvas = max(1, int(round(altura_espectro * escala_canvas)))
    largura_canvas = max(1, int(round(largura_espectro * escala_canvas)))

    imagem_base = Image.fromarray(imagem_espectro_vis).convert("RGB").resize(
        (largura_canvas, altura_canvas), Image.Resampling.LANCZOS
    )

    cor_pincel = {
        "Preto": "#000000",
        "Branco": "#ffffff",
        "Cinza suave": (
            f"rgb({int(intensidade * 255)} ,"
            f"{int(intensidade * 255)} ,"
            f"{int(intensidade * 255)})"
        ),
    }[tipo_pincel]

    canvas = st_canvas(
        background_image=imagem_base,
        drawing_mode="freedraw",
        stroke_width=tamanho_pincel,
        stroke_color=cor_pincel,
        update_streamlit=True,
        height=altura_canvas,
        width=largura_canvas,
        key="fourier_canvas",
    )

    col_ap, col_res = st.columns(2)
    with col_ap:
        aplicar = st.button(
            "✅ Aplicar edição e calcular inversa",
            key="fourier_aplicar",
        )
    with col_res:
        restaurar = st.button(
            "🔄 Restaurar espectro original",
            key="fourier_restaurar",
        )

    if restaurar:
        st.session_state["fourier_espectro_editado"] = espectro_original.copy()
        st.session_state.pop("fourier_resultado", None)
        st.rerun()

    if aplicar:
        if canvas.image_data is None:
            st.warning("Faça ao menos um traço no espectro antes de aplicar a edição.")
        else:
            novo_espectro = freq.aplicar_edicao(
                espectro_original,
                canvas.image_data,
                imagem_base,
                tipo_pincel,
                intensidade,
            )
            st.session_state["fourier_espectro_editado"] = novo_espectro

            with st.spinner("Calculando transformada inversa..."):
                resultado_float = freq.calcular_inversa(
                    novo_espectro,
                    metodo_atual,
                    h_orig,
                    w_orig,
                )

            resultado = freq.imagem_para_exibicao(resultado_float)
            st.session_state["fourier_resultado"] = resultado
            st.rerun()

    # ------------------------------------------------------------
    # 4. Resultado e download
    # ------------------------------------------------------------
    if "fourier_resultado" in st.session_state:
        st.divider()
        st.subheader("🖼️ Imagem filtrada (após transformada inversa)")

        resultado = st.session_state["fourier_resultado"]
        st.image(resultado, clamp=True, use_container_width=True)

        buffer = io.BytesIO()
        modo_saida = "L" if resultado.ndim == 2 else "RGB"
        Image.fromarray(resultado, mode=modo_saida).save(buffer, format="PNG")

        st.download_button(
            label="⬇️ Baixar imagem filtrada",
            data=buffer.getvalue(),
            file_name="imagem_filtrada_fourier.png",
            mime="image/png",
            key="fourier_download",
        )