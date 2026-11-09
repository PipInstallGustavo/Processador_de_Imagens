import numpy as np
from PIL import Image
import math

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

def normalizar_exibicao(valores: np.ndarray) -> Image.Image:
    minimo = valores.min()
    maximo = valores.max()
    if minimo == maximo:
        normalizados = np.zeros_like(valores)
    else:
        normalizados = (valores - minimo) * 255 / (maximo - minimo)
    return Image.fromarray(normalizados.astype(np.uint8))

def convolucao_numpy(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img_h, img_w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    resultado = np.zeros_like(img, dtype=np.float32)
    kernel = np.flipud(np.fliplr(kernel))
    
    for y in range(kh):
        for x in range(kw):
            resultado += padded[y:y+img_h, x:x+img_w] * kernel[y, x]
    return resultado

@process_channels
def aplicar_filtro_generico(img: Image.Image, kernel: np.ndarray, normalizar: bool = False) -> Image.Image:
    img_arr = np.array(img.convert('L'), dtype=float)
    res = convolucao_numpy(img_arr, kernel)
    if normalizar:
        return normalizar_exibicao(res)
    else:
        return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))

def filtro_media(img: Image.Image, tamanho: int = 3) -> Image.Image:
    kernel = np.ones((tamanho, tamanho), dtype=float) / (tamanho * tamanho)
    return aplicar_filtro_generico(img, kernel)

def filtro_media_ponderada(img: Image.Image, tamanho: int = 3) -> Image.Image:
    kernel = np.array([1.0, 2.0, 1.0])
    while len(kernel) < tamanho:
        kernel = np.convolve(kernel, [1.0, 2.0, 1.0])
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return aplicar_filtro_generico(img, kernel)

def filtro_gaussiano(img: Image.Image, tamanho: int = 3, sigma: float = 1.0) -> Image.Image:
    eixo = np.arange(-(tamanho // 2), tamanho // 2 + 1)
    x, y = np.meshgrid(eixo, eixo)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return aplicar_filtro_generico(img, kernel)

@process_channels
def filtro_mediana(img: Image.Image, tamanho: int = 3) -> Image.Image:
    imagem = np.array(img.convert("L"), dtype=float)
    raio = tamanho // 2
    padded = np.pad(imagem, raio, mode="reflect")
    altura, largura = imagem.shape
    resultado = np.zeros_like(imagem)
    for y in range(altura):
        for x in range(largura):
            vizinhanca = padded[y:y + tamanho, x:x + tamanho]
            resultado[y, x] = np.median(vizinhanca)
    return Image.fromarray(resultado.astype(np.uint8))

def filtro_laplaciano(img: Image.Image, normalizar: bool = True) -> Image.Image:
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=float)
    return aplicar_filtro_generico(img, kernel, normalizar=normalizar)

@process_channels
def aguacamento_laplaciano(img: Image.Image) -> Image.Image:
    imagem = np.array(img.convert("L"), dtype=float)
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)
    laplaciano = convolucao_numpy(imagem, kernel)
    resultado = imagem + laplaciano
    return Image.fromarray(np.clip(resultado, 0, 255).astype(np.uint8))

@process_channels
def filtro_high_boost(img: Image.Image, k: float = 1.5, tamanho: int = 3, sigma: float = 1.0) -> Image.Image:
    imagem = np.array(img.convert("L"), dtype=float)
    eixo = np.arange(-(tamanho // 2), tamanho // 2 + 1)
    x, y = np.meshgrid(eixo, eixo)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    suavizada = convolucao_numpy(imagem, kernel)
    mascara = imagem - suavizada
    resultado = imagem + k * mascara
    return Image.fromarray(np.clip(resultado, 0, 255).astype(np.uint8))

def filtro_sobel_x(img: Image.Image, normalizar: bool = True) -> Image.Image:
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    return aplicar_filtro_generico(img, kernel, normalizar=normalizar)

def filtro_sobel_y(img: Image.Image, normalizar: bool = True) -> Image.Image:
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    return aplicar_filtro_generico(img, kernel, normalizar=normalizar)

@process_channels
def gradiente_magnitude(img: Image.Image, normalizar: bool = True) -> Image.Image:
    imagem = np.array(img.convert("L"), dtype=float)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gx = convolucao_numpy(imagem, kx)
    gy = convolucao_numpy(imagem, ky)
    mag = np.sqrt(gx**2 + gy**2)
    if normalizar:
        return normalizar_exibicao(mag)
    return Image.fromarray(np.clip(mag, 0, 255).astype(np.uint8))

@process_channels
def intensificacao_fuzzy(img: Image.Image) -> Image.Image:
    imagem = np.array(img.convert("L"), dtype=float) / 255.0
    resultado = np.where(imagem <= 0.5, 2 * imagem**2, 1 - 2 * (1 - imagem)**2)
    return Image.fromarray((resultado * 255).astype(np.uint8))

@process_channels
def escala_vizinho(img: Image.Image, sx: float, sy: float) -> Image.Image:
    img_arr = np.array(img.convert('L'))
    h, w = img_arr.shape[:2]
    new_w, new_h = max(1, int(w * sx)), max(1, int(h * sy))
    
    x_new = np.arange(new_w)
    y_new = np.arange(new_h)
    
    x_orig = np.clip(np.round(x_new / sx).astype(int), 0, w - 1)
    y_orig = np.clip(np.round(y_new / sy).astype(int), 0, h - 1)
    
    X, Y = np.meshgrid(x_orig, y_orig)
    nova_img = img_arr[Y, X]
    return Image.fromarray(nova_img)

@process_channels
def escala_bilinear(img: Image.Image, sx: float, sy: float) -> Image.Image:
    img_arr = np.array(img.convert('L'), dtype=float)
    h, w = img_arr.shape
    new_w, new_h = max(1, int(w * sx)), max(1, int(h * sy))
    
    x_new = np.arange(new_w)
    y_new = np.arange(new_h)
    
    x = x_new / sx
    y = y_new / sy
    
    X, Y = np.meshgrid(x, y)
    
    X1 = np.floor(X).astype(int)
    Y1 = np.floor(Y).astype(int)
    X2 = np.clip(X1 + 1, 0, w - 1)
    Y2 = np.clip(Y1 + 1, 0, h - 1)
    X1 = np.clip(X1, 0, w - 1)
    Y1 = np.clip(Y1, 0, h - 1)
    
    DX = X - X1
    DY = Y - Y1
    
    Q11 = img_arr[Y1, X1]
    Q21 = img_arr[Y1, X2]
    Q12 = img_arr[Y2, X1]
    Q22 = img_arr[Y2, X2]
    
    R1 = Q11 * (1 - DX) + Q21 * DX
    R2 = Q12 * (1 - DX) + Q22 * DX
    P = R1 * (1 - DY) + R2 * DY
    
    return Image.fromarray(np.clip(P, 0, 255).astype(np.uint8))

@process_channels
def rotacao_vizinho(img: Image.Image, angulo: float, pivo_x: float = None, pivo_y: float = None) -> Image.Image:
    img_arr = np.array(img.convert('L'))
    h, w = img_arr.shape
    
    if pivo_x is None: pivo_x = (w - 1) / 2
    if pivo_y is None: pivo_y = (h - 1) / 2
        
    rad = math.radians(angulo)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)
    
    nova_img = np.zeros_like(img_arr)
    
    x_new, y_new = np.meshgrid(np.arange(w), np.arange(h))
    
    x = x_new - pivo_x
    y = y_new - pivo_y
    
    x_orig = x * cos_theta + y * sin_theta + pivo_x
    y_orig = -x * sin_theta + y * cos_theta + pivo_y
    
    x_orig = np.round(x_orig).astype(int)
    y_orig = np.round(y_orig).astype(int)
    
    valid = (x_orig >= 0) & (x_orig < w) & (y_orig >= 0) & (y_orig < h)
    nova_img[valid] = img_arr[y_orig[valid], x_orig[valid]]
    
    return Image.fromarray(nova_img)

@process_channels
def rotacao_bilinear(img: Image.Image, angulo: float, pivo_x: float = None, pivo_y: float = None) -> Image.Image:
    img_arr = np.array(img.convert('L'), dtype=float)
    h, w = img_arr.shape
    
    if pivo_x is None: pivo_x = (w - 1) / 2
    if pivo_y is None: pivo_y = (h - 1) / 2
        
    rad = math.radians(angulo)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)
    
    nova_img = np.zeros_like(img_arr)
    
    x_new, y_new = np.meshgrid(np.arange(w), np.arange(h))
    
    x = x_new - pivo_x
    y = y_new - pivo_y
    
    x_orig = x * cos_theta + y * sin_theta + pivo_x
    y_orig = -x * sin_theta + y * cos_theta + pivo_y
    
    valid = (x_orig >= 0) & (x_orig < w - 1) & (y_orig >= 0) & (y_orig < h - 1)
    
    X1 = np.floor(x_orig[valid]).astype(int)
    Y1 = np.floor(y_orig[valid]).astype(int)
    X2 = X1 + 1
    Y2 = Y1 + 1
    
    DX = x_orig[valid] - X1
    DY = y_orig[valid] - Y1
    
    Q11 = img_arr[Y1, X1]
    Q21 = img_arr[Y1, X2]
    Q12 = img_arr[Y2, X1]
    Q22 = img_arr[Y2, X2]
    
    R1 = Q11 * (1 - DX) + Q21 * DX
    R2 = Q12 * (1 - DX) + Q22 * DX
    P = R1 * (1 - DY) + R2 * DY
    
    nova_img[valid] = P
    
    return Image.fromarray(np.clip(nova_img, 0, 255).astype(np.uint8))

def demonstrar_aliasing(img: Image.Image, fator: int) -> Image.Image:
    """
    Demonstra o efeito de aliasing/Moiré:
    1. Subamostra a imagem (pega 1 pixel a cada `fator` pixels)
    2. Reconstrói para o tamanho original via vizinho mais próximo
    Quanto maior o fator, mais visível o efeito.
    """
    arr = np.array(img)
    subamostrada = arr[::fator, ::fator]
    # Reconstrói para o tamanho original
    h_orig, w_orig = arr.shape[:2]
    img_sub = Image.fromarray(subamostrada)
    img_rec = img_sub.resize((w_orig, h_orig), Image.Resampling.NEAREST)
    return img_rec
