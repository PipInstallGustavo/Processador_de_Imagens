import numpy as np
from PIL import Image
import math
from .intensidade import equalizacao_histograma as eq_hist_gray

def rgb_to_hsi(img_array: np.ndarray) -> np.ndarray:
    # Retorna hsi (H: 0-360, S: 0-1, I: 0-255)
    img_float = img_array.astype(np.float32) / 255.0
    R, G, B = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
    
    I = (R + G + B) / 3.0
    
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.zeros_like(I)
    I_nz = I > 0
    S[I_nz] = 1 - min_rgb[I_nz] / I[I_nz]
    
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1, 1))
    
    H = theta.copy()
    H[B > G] = 2 * np.pi - H[B > G]
    
    H = np.degrees(H) # 0 a 360
    
    HSI = np.zeros_like(img_float)
    HSI[:,:,0] = H
    HSI[:,:,1] = S
    HSI[:,:,2] = I * 255
    return HSI

def hsi_to_rgb(hsi_array: np.ndarray) -> np.ndarray:
    H = np.radians(hsi_array[:,:,0])
    S = hsi_array[:,:,1]
    I = hsi_array[:,:,2] / 255.0
    
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    
    # Setor RG (0 <= H < 120)
    idx = (H >= 0) & (H < 2*np.pi/3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi/3 - H[idx]))
    G[idx] = 3*I[idx] - (R[idx] + B[idx])
    
    # Setor GB (120 <= H < 240)
    idx = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    H_gb = H[idx] - 2*np.pi/3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H_gb) / np.cos(np.pi/3 - H_gb))
    B[idx] = 3*I[idx] - (R[idx] + G[idx])
    
    # Setor BR (240 <= H <= 360)
    idx = (H >= 4*np.pi/3) & (H <= 2*np.pi)
    H_br = H[idx] - 4*np.pi/3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H_br) / np.cos(np.pi/3 - H_br))
    R[idx] = 3*I[idx] - (G[idx] + B[idx])
    
    RGB = np.stack((R, G, B), axis=-1)
    return np.clip(RGB * 255, 0, 255).astype(np.uint8)

# def escala_cinza_media(img: Image.Image) -> Image.Image:
#     arr = np.array(img).astype(np.float32)
#     gray = arr.mean(axis=2)
#     return Image.fromarray(gray.astype(np.uint8))

def escala_cinza_media(imagem: Image.Image) -> Image.Image:
    arr = np.array(imagem)

    # Se a imagem já estiver em escala de cinza, não precisa converter.
    if arr.ndim == 2:
        return imagem.convert("L")

    # Calcula a média dos canais RGB.
    gray = arr[:, :, :3].mean(axis=2)

    return Image.fromarray(gray.astype(np.uint8))

def escala_cinza_ponderada(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    # Ponderação típica do NTSC
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    return Image.fromarray(gray.astype(np.uint8))


def rgb_para_hsv(r, g, b):
    # Normaliza RGB para [0, 1].
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    # Value.
    v = c_max

    # Saturation.
    if c_max == 0:
        s = 0
    else:
        s = delta / c_max

    # Hue.
    if delta == 0:
        h = 0
    elif c_max == r:
        h = 60 * (((g - b) / delta) % 6)
    elif c_max == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)

    return h, s, v


def chroma_key(imagem: Image.Image, fundo: Image.Image, tolerancia: int = 20, caminho_saida=None) -> Image.Image:
    # Recebe imagens já abertas e converte para RGB.
    imagem = imagem.convert("RGB")
    fundo = fundo.convert("RGB")

    largura, altura = imagem.size

    # Ajusta o fundo às dimensões da imagem original.
    fundo = fundo.resize((largura, altura))

    # Cria a imagem de saída.
    resultado = Image.new("RGB", (largura, altura))

    pixels = imagem.load()
    pixels_fundo = fundo.load()
    pixels_saida = resultado.load()

    # Define a faixa de tons verdes a partir da tolerância.
    h_min = max(0, 70 - tolerancia)
    h_max = min(360, 140 + tolerancia)

    for y in range(altura):
        for x in range(largura):

            # Obtém o pixel original.
            r, g, b = pixels[x, y]

            # Conversão RGB -> HSV implementada manualmente.
            h, s, v = rgb_para_hsv(r, g, b)

            # Identifica os tons de verde dentro da tolerância.
            if h_min <= h <= h_max and s >= 50 / 255 and v >= 50 / 255:

                # Substitui o verde pelo pixel da imagem de fundo.
                pixels_saida[x, y] = pixels_fundo[x, y]

            else:
                # Preserva o pixel original.
                pixels_saida[x, y] = (r, g, b)

    # Salva a imagem se um caminho tiver sido informado.
    if caminho_saida is not None:
        resultado.save(caminho_saida)

    return resultado


def equalizar_hsi(img: Image.Image) -> Image.Image:
    hsi = rgb_to_hsi(np.array(img))
    I = hsi[:,:,2]
    
    # Equaliza só a intensidade
    img_I = Image.fromarray(I.astype(np.uint8))
    img_I_eq = eq_hist_gray(img_I)
    
    hsi[:,:,2] = np.array(img_I_eq)
    rgb = hsi_to_rgb(hsi)
    return Image.fromarray(rgb)

def ajustar_hsi(img: Image.Image, h_shift: float, s_scale: float, i_shift: float) -> Image.Image:
    hsi = rgb_to_hsi(np.array(img))
    
    hsi[:,:,0] = (hsi[:,:,0] + h_shift) % 360
    hsi[:,:,1] = np.clip(hsi[:,:,1] * s_scale, 0, 1)
    hsi[:,:,2] = np.clip(hsi[:,:,2] + i_shift, 0, 255)
    
    rgb = hsi_to_rgb(hsi)
    return Image.fromarray(rgb)

def ajustar_canais(img: Image.Image, r: float=1, g: float=1, b: float=1) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * r, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1] * g, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] * b, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def sepia(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    tr = 0.393 * R + 0.769 * G + 0.189 * B
    tg = 0.349 * R + 0.686 * G + 0.168 * B
    tb = 0.272 * R + 0.534 * G + 0.131 * B
    
    res = np.stack((tr, tg, tb), axis=2)
    return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))

def visualizar_hsi(img: Image.Image) -> Image.Image:
    """
    Converte RGB → HSI e retorna uma imagem composta que mostra os
    três canais lado a lado: H (normalizado 0-255), S (0-255), I (0-255).
    Serve como a 'ferramenta de transformação RGB ↔ HSI' exigida pelo trabalho.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    arr = np.array(img.convert('RGB'))
    hsi = rgb_to_hsi(arr)  # H: 0-360, S: 0-1, I: 0-255

    H_norm = (hsi[:, :, 0] / 360.0 * 255).astype(np.uint8)
    S_norm = (hsi[:, :, 1] * 255).astype(np.uint8)
    I_norm = hsi[:, :, 2].astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Canais HSI (RGB \u2192 HSI)', fontsize=14)

    for ax, canal, titulo, cmap in zip(
        axes,
        [H_norm, S_norm, I_norm],
        ['Matiz (H) — 0=vermelho, 120=verde, 240=azul',
         'Satura\u00e7\u00e3o (S)',
         'Intensidade (I)'],
        ['hsv', 'gray', 'gray']
    ):
        ax.imshow(canal, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(titulo)
        ax.axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    return Image.open(buf).copy()

def chroma_key_numpy(frente_rgb: np.ndarray, fundo_rgb: np.ndarray,
                     cor_key=(0, 255, 0), tolerancia: int = 20) -> np.ndarray:
    """
    Versão vetorizada do Chroma-Key que opera diretamente em arrays NumPy.
    Usada pelo stream de vídeo ao vivo para evitar conversões PIL frame a frame.
    """
    # Converte a cor-chave para HSV escalar
    r_k, g_k, b_k = cor_key
    h_key, _, _ = rgb_para_hsv(r_k, g_k, b_k)  # h_key em graus [0-360]

    # Converte cada pixel da frente para HSV (vetorizado via rgb_to_hsi aprox)
    # Usamos a função rgb_to_hsi que já temos
    hsi = rgb_to_hsi(frente_rgb)  # H: 0-360
    H = hsi[:, :, 0]
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]  # 0-255

    # Faixa de matiz em torno da cor-chave
    h_min = max(0, h_key - tolerancia * 2)
    h_max = min(360, h_key + tolerancia * 2)

    mascara = (H >= h_min) & (H <= h_max) & (S >= 50/255) & (I >= 50)

    resultado = frente_rgb.copy()
    resultado[mascara] = fundo_rgb[mascara]
    return resultado
