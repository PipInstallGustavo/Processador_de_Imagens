import io
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def process_channels(func):
    def wrapper(img: Image.Image, *args, **kwargs):
        if img.mode == 'RGB':
            r, g, b = img.split()
            r_res = func(r, *args, **kwargs)
            g_res = func(g, *args, **kwargs)
            b_res = func(b, *args, **kwargs)
            return Image.merge('RGB', (r_res, g_res, b_res))
        return func(img, *args, **kwargs)
    return wrapper

@process_channels
def limiarizacao(img: Image.Image, limiar: float = 0.5) -> Image.Image:
    img = img.convert('L')
    img_array = np.array(img)
    limiar_val = limiar * 255 if limiar <= 1.0 else limiar
    img_array = np.where(img_array < limiar_val, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def negativo(img: Image.Image) -> Image.Image:
    if img.mode != 'L' and img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    img_array = 255 - img_array
    return Image.fromarray(img_array)

@process_channels
def brilho(img: Image.Image, valor: int = 50) -> Image.Image:
    img_array = np.array(img).astype(np.int16)
    img_array = img_array + valor
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

@process_channels
def correcao_gama(img: Image.Image, c: float = 1.0, gama: float = 1.0) -> Image.Image:
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = c * (img_array ** gama)
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

@process_channels
def contrast_stretching(img: Image.Image, rmin: int, rmax: int) -> Image.Image:
    img_array = np.array(img.convert('L')).astype(np.float32)
    s = np.zeros_like(img_array)
    mask_mid = (img_array >= rmin) & (img_array <= rmax)
    mask_high = (img_array > rmax)
    
    s[mask_mid] = (img_array[mask_mid] - rmin) * 255 / (rmax - rmin)
    s[mask_high] = 255
    s[img_array < rmin] = 0
    
    return Image.fromarray(np.clip(s, 0, 255).astype(np.uint8))

def linear_por_partes(img: Image.Image, pontos: list) -> Image.Image:
    """
    Transformação linear por partes genérica.
    pontos: lista de (r_entrada, s_saida) ordenados por r_entrada.
    Interpola linearmente entre cada par de breakpoints.
    Funciona com imagens coloridas (aplica canal a canal) e em cinza.
    """
    # Garante que os extremos estão presentes
    pontos = sorted(pontos, key=lambda p: p[0])
    if pontos[0][0] > 0:
        pontos = [(0, 0)] + pontos
    if pontos[-1][0] < 255:
        pontos = pontos + [(255, 255)]

    # Cria a LUT (256 valores)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(len(pontos) - 1):
        r0, s0 = pontos[i]
        r1, s1 = pontos[i + 1]
        for r in range(r0, r1 + 1):
            if r1 == r0:
                lut[r] = s0
            else:
                lut[r] = int(s0 + (s1 - s0) * (r - r0) / (r1 - r0))

    arr = np.array(img)
    resultado = lut[arr]  # aplica a LUT em todos os canais automaticamente
    return Image.fromarray(resultado.astype(np.uint8))

def esteganografia_esconder(img: Image.Image, mensagem: str) -> Image.Image:
    img = img.copy()
    img_array = np.array(img)
    # int16 evita OverflowError no operador ~1 com dtype uint8
    flat_img = img_array.flatten().astype(np.int16)

    msg_bin = ''.join([format(ord(c), '08b') for c in mensagem])
    msg_bin += '00000000'

    if len(msg_bin) > len(flat_img):
        raise ValueError("Mensagem muito longa para essa imagem.")

    for i in range(len(msg_bin)):
        bit = int(msg_bin[i])
        flat_img[i] = (flat_img[i] & ~1) | bit

    res_array = flat_img.astype(np.uint8).reshape(img_array.shape)
    return Image.fromarray(res_array)

def esteganografia_revelar(img: Image.Image) -> str:
    flat_img = np.array(img).flatten()
    msg_bin = ""
    for pixel in flat_img:
        msg_bin += str(pixel & 1)
        if len(msg_bin) % 8 == 0:
            if msg_bin[-8:] == '00000000':
                break
                
    chars = []
    for i in range(0, len(msg_bin) - 8, 8):
        byte = msg_bin[i:i+8]
        chars.append(chr(int(byte, 2)))
        
    return "".join(chars)

def plotar_histograma(img: Image.Image) -> Image.Image:
    """
    Para imagens coloridas: exibe histogramas separados de R, G, B e I (intensidade).
    Para imagens em escala de cinza: exibe histograma único.
    """
    arr = np.array(img)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Imagem colorida — 4 subplots: R, G, B, I
        R = arr[:, :, 0].flatten()
        G = arr[:, :, 1].flatten()
        B = arr[:, :, 2].flatten()
        I = ((arr[:, :, 0].astype(float) + arr[:, :, 1] + arr[:, :, 2]) / 3).flatten()

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle('Histograma da Imagem (RGB + Intensidade)', fontsize=14)

        for ax, canal, cor, titulo in zip(
            axes.flat,
            [R, G, B, I],
            ['red', 'green', 'blue', 'gray'],
            ['Canal R (Vermelho)', 'Canal G (Verde)', 'Canal B (Azul)', 'Intensidade (I)']
        ):
            ax.hist(canal, bins=256, range=(0, 256), color=cor, alpha=0.75)
            ax.set_title(titulo)
            ax.set_xlabel('Intensidade')
            ax.set_ylabel('Nº de Pixels')
            ax.set_xlim(0, 255)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
    else:
        # Escala de cinza
        pixels = arr.flatten()
        plt.figure(figsize=(12, 6))
        plt.hist(pixels, bins=256, range=(0, 256), color='gray', alpha=0.75)
        plt.title('Histograma da Imagem (Escala de Cinza)')
        plt.xlabel('Intensidade de Cinza')
        plt.ylabel('Número de Pixels')
        plt.xlim(0, 255)
        plt.grid(axis='y', alpha=0.3)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='PNG', bbox_inches='tight')
    plt.close('all')
    buffer.seek(0)
    return Image.open(buffer).copy()

def equalizacao_histograma(img: Image.Image) -> Image.Image:
    is_rgb = img.mode == 'RGB'
    if is_rgb:
        # Converter para HSV, equalizar o V, e voltar para RGB
        import matplotlib.colors as mcolors
        hsv = mcolors.rgb_to_hsv(np.array(img) / 255.0)
        v = (hsv[:,:,2] * 255).astype(np.uint8)
    else:
        v = np.array(img.convert('L'))

    hist, _ = np.histogram(v.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf = np.ma.filled(cdf_masked, 0).astype('uint8')
    
    v_eq = cdf[v]
    
    if is_rgb:
        hsv[:,:,2] = v_eq / 255.0
        rgb = (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
        return Image.fromarray(rgb)
    else:
        return Image.fromarray(v_eq)
