"""Transformadas de Fourier para o Processador de Imagens.

Suporta imagens em escala de cinza (H, W) e RGB (H, W, 3).

Métodos disponíveis:
- DFT/IDFT 2D manual (DFT separável, com limite recomendado de tamanho);
- FFT/IFFT 2D manual pelo algoritmo Cooley-Tukey radix-2;
- FFT/IFFT do NumPy como alternativa rápida.

A FFT manual aplica zero-padding até a próxima potência de 2. A imagem
reconstruída é recortada para suas dimensões originais.
"""

import math
import numpy as np
from PIL import Image


# ============================================================
# 1. DFT / IDFT 1D E 2D MANUAIS
# ============================================================

def dft_1d(sinal: np.ndarray) -> np.ndarray:
    """DFT 1D direta, calculada explicitamente. Complexidade O(N²)."""
    sinal = np.asarray(sinal, dtype=complex)
    n = sinal.size
    if n == 0:
        raise ValueError("O sinal não pode ser vazio.")

    k = np.arange(n)
    t = np.arange(n)
    matriz = np.exp(-2j * np.pi * np.outer(k, t) / n)
    return matriz @ sinal


def idft_1d(espectro: np.ndarray) -> np.ndarray:
    """IDFT 1D direta, com normalização por N."""
    espectro = np.asarray(espectro, dtype=complex)
    n = espectro.size
    if n == 0:
        raise ValueError("O espectro não pode ser vazio.")

    k = np.arange(n)
    t = np.arange(n)
    matriz = np.exp(2j * np.pi * np.outer(k, t) / n)
    return (matriz @ espectro) / n


def _validar_imagem(imagem: np.ndarray) -> np.ndarray:
    """Aceita somente imagem 2D (cinza) ou 3D com 3 canais (RGB)."""
    imagem = np.asarray(imagem, dtype=float)
    if imagem.ndim == 2:
        pass
    elif imagem.ndim == 3 and imagem.shape[2] == 3:
        pass
    else:
        raise ValueError(
            "A imagem deve ter formato (altura, largura) ou (altura, largura, 3)."
        )
    if imagem.shape[0] == 0 or imagem.shape[1] == 0:
        raise ValueError("A imagem não pode ter dimensões vazias.")
    return imagem


def _aplicar_por_canal(imagem: np.ndarray, funcao):
    """Aplica uma transformada 2D separadamente em cada canal, se RGB."""
    imagem = _validar_imagem(imagem)
    if imagem.ndim == 2:
        return funcao(imagem)
    canais = [funcao(imagem[:, :, c]) for c in range(3)]
    return np.stack(canais, axis=2)


def _dft_2d_cinza(imagem: np.ndarray) -> np.ndarray:
    altura, largura = imagem.shape
    temporaria = np.zeros((altura, largura), dtype=complex)
    resultado = np.zeros((altura, largura), dtype=complex)

    for y in range(altura):
        temporaria[y, :] = dft_1d(imagem[y, :])
    for x in range(largura):
        resultado[:, x] = dft_1d(temporaria[:, x])
    return resultado


def dft_2d(imagem: np.ndarray) -> np.ndarray:
    """DFT 2D manual; aceita escala de cinza ou RGB."""
    return _aplicar_por_canal(imagem, _dft_2d_cinza)


def _idft_2d_cinza(espectro: np.ndarray) -> np.ndarray:
    altura, largura = espectro.shape
    temporaria = np.zeros((altura, largura), dtype=complex)
    resultado = np.zeros((altura, largura), dtype=complex)

    for y in range(altura):
        temporaria[y, :] = idft_1d(espectro[y, :])
    for x in range(largura):
        resultado[:, x] = idft_1d(temporaria[:, x])
    return resultado.real


def idft_2d(espectro: np.ndarray) -> np.ndarray:
    """IDFT 2D manual; aceita espectros em escala de cinza ou RGB."""
    espectro = np.asarray(espectro, dtype=complex)
    if espectro.ndim == 2:
        return _idft_2d_cinza(espectro)
    if espectro.ndim == 3 and espectro.shape[2] == 3:
        return np.stack(
            [_idft_2d_cinza(espectro[:, :, c]) for c in range(3)], axis=2
        )
    raise ValueError("O espectro deve ser 2D ou possuir 3 canais RGB.")


# ============================================================
# 2. FFT / IFFT MANUAIS — COOLEY-TUKEY RADIX-2
# ============================================================

def _proxima_pot2(n: int) -> int:
    """Menor potência de 2 maior ou igual a n."""
    if n < 1:
        raise ValueError("A dimensão deve ser positiva.")
    return 1 << (n - 1).bit_length()


def fft_1d(sinal: np.ndarray) -> np.ndarray:
    """FFT Cooley-Tukey recursiva. O comprimento deve ser potência de 2."""
    sinal = np.asarray(sinal, dtype=complex)
    n = sinal.size
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("O tamanho do sinal deve ser uma potência de 2.")
    if n == 1:
        return sinal.copy()

    pares = fft_1d(sinal[0::2])
    impares = fft_1d(sinal[1::2])
    fatores = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    metade = fatores * impares
    return np.concatenate((pares + metade, pares - metade))


def ifft_1d(espectro: np.ndarray) -> np.ndarray:
    """IFFT 1D pela relação com a FFT do conjugado."""
    espectro = np.asarray(espectro, dtype=complex)
    if espectro.size == 0:
        raise ValueError("O espectro não pode ser vazio.")
    return np.conjugate(fft_1d(np.conjugate(espectro))) / espectro.size


def _fft_2d_cinza(imagem: np.ndarray):
    h_original, w_original = imagem.shape
    altura = _proxima_pot2(h_original)
    largura = _proxima_pot2(w_original)

    # Preenchimento com zeros: não altera/interpola os pixels existentes.
    preenchida = np.zeros((altura, largura), dtype=complex)
    preenchida[:h_original, :w_original] = imagem

    temporaria = np.zeros_like(preenchida)
    resultado = np.zeros_like(preenchida)
    for y in range(altura):
        temporaria[y, :] = fft_1d(preenchida[y, :])
    for x in range(largura):
        resultado[:, x] = fft_1d(temporaria[:, x])
    return resultado


def fft_2d(imagem: np.ndarray):
    """FFT 2D manual para cinza/RGB. Retorna (espectro, h_original, w_original)."""
    imagem = _validar_imagem(imagem)
    h_original, w_original = imagem.shape[:2]
    if imagem.ndim == 2:
        espectro = _fft_2d_cinza(imagem)
    else:
        espectro = np.stack(
            [_fft_2d_cinza(imagem[:, :, c]) for c in range(3)], axis=2
        )
    return espectro, h_original, w_original


def _ifft_2d_cinza(espectro: np.ndarray) -> np.ndarray:
    altura, largura = espectro.shape
    temporaria = np.zeros((altura, largura), dtype=complex)
    resultado = np.zeros((altura, largura), dtype=complex)

    for y in range(altura):
        temporaria[y, :] = ifft_1d(espectro[y, :])
    for x in range(largura):
        resultado[:, x] = ifft_1d(temporaria[:, x])
    return resultado.real


def ifft_2d(espectro: np.ndarray, h_orig: int, w_orig: int) -> np.ndarray:
    """IFFT 2D manual e recorte para as dimensões originais (cinza/RGB)."""
    espectro = np.asarray(espectro, dtype=complex)
    if espectro.ndim == 2:
        imagem = _ifft_2d_cinza(espectro)
        return imagem[:h_orig, :w_orig]
    if espectro.ndim == 3 and espectro.shape[2] == 3:
        canais = [
            _ifft_2d_cinza(espectro[:, :, c])[:h_orig, :w_orig]
            for c in range(3)
        ]
        return np.stack(canais, axis=2)
    raise ValueError("O espectro deve ser 2D ou possuir 3 canais RGB.")


# ============================================================
# 3. FFTSHIFT / IFFTSHIFT MANUAIS
# ============================================================

def fftshift_manual(espectro: np.ndarray) -> np.ndarray:
    """Centraliza a frequência zero; em RGB, desloca apenas os eixos H e W."""
    espectro = np.asarray(espectro)
    if espectro.ndim not in (2, 3):
        raise ValueError("O espectro deve ser 2D ou 3D RGB.")
    return np.roll(
        np.roll(espectro, espectro.shape[0] // 2, axis=0),
        espectro.shape[1] // 2, axis=1
    )


def ifftshift_manual(espectro: np.ndarray) -> np.ndarray:
    """Desfaz fftshift, inclusive para dimensões ímpares."""
    espectro = np.asarray(espectro)
    if espectro.ndim not in (2, 3):
        raise ValueError("O espectro deve ser 2D ou 3D RGB.")
    return np.roll(
        np.roll(espectro, -(espectro.shape[0] // 2), axis=0),
        -(espectro.shape[1] // 2), axis=1
    )


# ============================================================
# 4. VISUALIZAÇÃO E EDIÇÃO DO ESPECTRO
# ============================================================

def _normalizar_uint8(matriz: np.ndarray) -> np.ndarray:
    matriz = np.asarray(matriz, dtype=float)
    minimo = np.nanmin(matriz)
    maximo = np.nanmax(matriz)
    if not np.isfinite(minimo) or not np.isfinite(maximo) or maximo <= minimo:
        return np.zeros(matriz.shape, dtype=np.uint8)
    normalizada = (matriz - minimo) / (maximo - minimo)
    return np.clip(normalizada * 255, 0, 255).astype(np.uint8)


def visualizar_espectro(espectro: np.ndarray) -> np.ndarray:
    """Gera a visualização logarítmica do espectro deslocado.

    Cinza: retorna (H, W). RGB: retorna (H, W, 3), normalizando cada canal
    separadamente para que as três componentes sejam visualizáveis.
    """
    espectro = np.asarray(espectro, dtype=complex)
    deslocado = fftshift_manual(espectro)
    magnitude_log = np.log1p(np.abs(deslocado))

    if magnitude_log.ndim == 2:
        return _normalizar_uint8(magnitude_log)
    return np.stack(
        [_normalizar_uint8(magnitude_log[:, :, c]) for c in range(3)], axis=2
    )


def _restaurar_simetria_conjugada(espectro: np.ndarray) -> np.ndarray:
    """Impõe F(-u,-v) = conjugado(F(u,v)) em cada canal."""
    espelhado = np.conjugate(
        np.roll(
            np.roll(espectro[::-1, ::-1, ...], 1, axis=0),
            1, axis=1
        )
    )
    return (espectro + espelhado) / 2


def _editar_espectro_cinza(espectro: np.ndarray, mascara: np.ndarray,
                            tipo_pincel: str, intensidade: float) -> np.ndarray:
    deslocado = fftshift_manual(espectro)
    magnitude = np.abs(deslocado).copy()
    fase = np.angle(deslocado)
    maximo = float(magnitude.max()) if magnitude.size else 0.0

    if tipo_pincel.lower() == "preto":
        alvo = 0.0
    elif tipo_pincel.lower() == "branco":
        alvo = maximo
    else:
        alvo = maximo * float(np.clip(intensidade, 0.0, 1.0))

    magnitude[mascara] = alvo
    magnitude = ifftshift_manual(magnitude)
    fase = ifftshift_manual(fase)
    novo_espectro = magnitude * np.exp(1j * fase)
    return _restaurar_simetria_conjugada(novo_espectro)


def aplicar_edicao(espectro: np.ndarray, imagem_canvas,
                   imagem_base: Image.Image = None,
                   tipo_pincel: str = "Preto",
                   intensidade: float = 0.5) -> np.ndarray:
    """Aplica os traços do st_canvas à magnitude do espectro.

    Aceita espectro 2D (cinza) ou (H, W, 3) (RGB). A máscara desenhada
    é aplicada igualmente aos três canais RGB. A fase é preservada e a
    simetria conjugada é restaurada para permitir reconstrução real.

    ``imagem_base`` é mantido por compatibilidade com a chamada anterior;
    a identificação dos traços é feita pelo canal alpha do canvas.
    """
    espectro = np.asarray(espectro, dtype=complex)
    if imagem_canvas is None:
        return espectro.copy()

    canvas = np.asarray(imagem_canvas)
    if canvas.ndim != 3 or canvas.shape[2] < 4:
        return espectro.copy()
    if espectro.ndim not in (2, 3) or (espectro.ndim == 3 and espectro.shape[2] != 3):
        raise ValueError("O espectro deve ser 2D ou possuir 3 canais RGB.")

    altura, largura = espectro.shape[:2]
    alpha = canvas[:, :, 3].astype(np.uint8)
    if not np.any(alpha > 0):
        return espectro.copy()

    mascara_pil = Image.fromarray(alpha).resize(
        (largura, altura), Image.Resampling.BILINEAR
    )
    mascara = np.asarray(mascara_pil) > 20

    if espectro.ndim == 2:
        return _editar_espectro_cinza(espectro, mascara, tipo_pincel, intensidade)

    canais_editados = [
        _editar_espectro_cinza(espectro[:, :, c], mascara, tipo_pincel, intensidade)
        for c in range(3)
    ]
    return np.stack(canais_editados, axis=2)


# ============================================================
# 5. API UNIFICADA — COMPATÍVEL COM A PÁGINA EXISTENTE
# ============================================================

def _redimensionar(imagem: np.ndarray, tamanho: tuple) -> np.ndarray:
    """Redimensiona imagem cinza ou RGB usando PIL."""
    imagem = _validar_imagem(imagem)
    if imagem.ndim == 2:
        pil = Image.fromarray(np.clip(imagem, 0, 255).astype(np.uint8), mode="L")
    else:
        pil = Image.fromarray(np.clip(imagem, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize(tamanho, Image.Resampling.LANCZOS), dtype=float)


def preparar_imagem(imagem: np.ndarray, metodo: str) -> np.ndarray:
    """Prepara a entrada para o método escolhido.

    DFT manual: limita a maior dimensão a 64 pixels para evitar custo
    computacional excessivo. FFT manual: não redimensiona; o zero-padding
    é feito dentro de fft_2d. FFT NumPy: mantém o tamanho original.
    """
    imagem = _validar_imagem(imagem)
    altura, largura = imagem.shape[:2]

    if metodo == "DFT manual":
        limite = 64
        escala = min(1.0, limite / max(altura, largura))
        if escala < 1.0:
            nova_largura = max(1, int(round(largura * escala)))
            nova_altura = max(1, int(round(altura * escala)))
            imagem = _redimensionar(imagem, (nova_largura, nova_altura))

    return imagem


def calcular_transformada(imagem: np.ndarray, metodo: str):
    """Calcula a transformada e retorna sempre (espectro, h_original, w_original).

    Para DFT manual, use preparar_imagem antes, se necessário, devido ao
    custo O(H²W + HW²). A FFT manual faz zero-padding internamente.
    """
    imagem = _validar_imagem(imagem)
    altura, largura = imagem.shape[:2]

    if metodo == "DFT manual":
        return dft_2d(imagem), altura, largura
    if metodo == "FFT manual":
        return fft_2d(imagem)
    if metodo in ("FFT NumPy", "NumPy", "FFT biblioteca"):
        eixos = (0, 1)
        espectro = np.fft.fft2(imagem, axes=eixos)
        return espectro, altura, largura
    raise ValueError(f"Método de transformada desconhecido: {metodo}")


def calcular_inversa(espectro: np.ndarray, metodo: str,
                     h_orig: int, w_orig: int) -> np.ndarray:
    """Calcula a transformada inversa; retorna cinza 2D ou RGB 3D em float."""
    espectro = np.asarray(espectro, dtype=complex)

    if metodo == "DFT manual":
        return idft_2d(espectro)
    if metodo == "FFT manual":
        return ifft_2d(espectro, h_orig, w_orig)
    if metodo in ("FFT NumPy", "NumPy", "FFT biblioteca"):
        imagem = np.fft.ifft2(espectro, axes=(0, 1)).real
        return imagem[:h_orig, :w_orig, ...]
    raise ValueError(f"Método de transformada desconhecido: {metodo}")


def imagem_para_exibicao(imagem: np.ndarray) -> np.ndarray:
    """Converte resultado float em uint8 para exibição, sem alterar o cálculo.

    Para imagens RGB, apenas limita cada componente ao intervalo [0, 255].
    Para imagens cinza, também limita ao intervalo [0, 255]."""
    imagem = np.asarray(imagem, dtype=float)
    return np.clip(np.rint(imagem), 0, 255).astype(np.uint8)