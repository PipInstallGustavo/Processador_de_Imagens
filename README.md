# Processador de Imagens

Aplicação web interativa de processamento de imagens digitais, construída com **Python** e **Streamlit**. Cobre todas as implementações, desde transformações básicas de intensidade até filtragem no domínio da frequência com edição interativa do espectro de Fourier.

---

## 📁 Estrutura do Projeto

```
processador_de_imagens/
│
├── app.py                      # Ponto de entrada principal (Streamlit)
│
├── modulos/                    # Implementações de cada área temática
│   ├── intensidade.py          # Transformações de intensidade e histogramas
│   ├── espacial.py             # Filtros espaciais e operações geométricas
│   ├── frequencia.py           # DFT/FFT manuais, edição de espectro
│   └── cores.py                # Processamento de imagens coloridas (HSI, Chroma-Key…)
│
├── paginas/                    # Páginas com lógica de UI mais complexa
│   └── fourier.py              # Página completa da Transformada de Fourier
├── imagens/                    # Pasta para salvar resultados (criada automaticamente)
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

---

## ⚙️ Como Rodar

### 1. Pré-requisitos

- Python 3.10 ou superior
- `pip`

### 2. Criar e ativar o ambiente virtual

```bash
# Na pasta raiz do projeto:
python3 -m venv venv_proc_imagens
source venv_proc_imagens/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Iniciar a aplicação

```bash
streamlit run app.py
```

O Streamlit abrirá automaticamente no navegador em `http://localhost:8501`.

> **Dica:** Deixe o terminal aberto enquanto usa a aplicação. Para encerrar, pressione `Ctrl+C`.

---

## 🧭 Como Usar

A interface é dividida em duas partes:

- **Barra Lateral (esquerda):** Menu de navegação, upload de imagem e controles de cada operação.
- **Área Principal (centro):** Exibe a imagem original à esquerda e o resultado processado à direita.

Selecione a categoria no topo da barra lateral, faça o upload de uma imagem e escolha a operação desejada. O resultado aparece imediatamente, e um botão **"Baixar Imagem Processada"** permite salvar o resultado em PNG.

---

## 📋 Funcionalidades Implementadas

### 🔆 Processamento de Intensidade

Todas as funções abaixo funcionam tanto para imagens em **escala de cinza** quanto **coloridas (RGB)** — para imagens coloridas, a operação é aplicada canal a canal (R, G, B).

| Operação | Descrição |
|----------|-----------|
| **Limiarização** | Converte para binário (preto/branco) com limiar ajustável (0–1) |
| **Negativo** | Inverte os valores de intensidade: `s = 255 - r` |
| **Brilho** | Soma uma constante a cada pixel (pode ser negativa para escurecer) |
| **Correção de Gama** | Transformação de potência: `s = c · rᵞ`, com `c` e `γ` ajustáveis |
| **Alargamento de Contraste** | Transformação linear por partes: estica o intervalo `[Rmin, Rmax]` para `[0, 255]` |
| **Histograma** | Exibe o histograma; para imagens coloridas mostra 4 gráficos: R, G, B e Intensidade |
| **Equalização de Histograma** | Redistribui as intensidades para um histograma mais uniforme (opera no canal I do HSI para imagens coloridas) |
| **Esteganografia (Esconder)** | Oculta uma mensagem de texto nos bits menos significativos (LSB) dos pixels |
| **Esteganografia (Revelar)** | Extrai e exibe a mensagem oculta por esteganografia LSB |

---

### 🧹 Filtragem e Operações Espaciais

Todas as operações de filtragem funcionam com imagens coloridas e em tons de cinza.

#### Filtros de Suavização
| Operação | Descrição |
|----------|-----------|
| **Filtro Genérico (Convolução)** | Aplica qualquer kernel definido pelo usuário. Suporta kernels ímpares: 3×3, 5×5, 7×7, 9×9… |
| **Média** | Kernel de média simples de tamanho N×N |
| **Média Ponderada** | Kernel Gaussiano binomial (ponderação pelo triângulo de Pascal) |
| **Gaussiano** | Kernel Gaussiano contínuo com sigma ajustável |
| **Mediana** | Filtro não-linear; excelente para remover ruído sal-e-pimenta |

> **Obs.:** A imagem processada mantém sempre o mesmo tamanho da original (`padding` nas bordas).

#### Filtros de Aguçamento
| Operação | Descrição |
|----------|-----------|
| **Laplaciano** | Detecta bordas via derivada de segunda ordem. Resultado normalizado para tons de cinza |
| **Aguçamento Laplaciano** | Soma o Laplaciano à imagem original para realçar detalhes |
| **High-Boost** | `resultado = original + k × (original − suavizada)`. Fator `k` ajustável |
| **Sobel X / Sobel Y** | Detectam bordas horizontais e verticais, respectivamente. Resultado normalizado |
| **Gradiente (Magnitude)** | Combina Sobel X e Y: `M = √(Gx² + Gy²)` |
| **Fuzzy** | Intensificação não-linear baseada em lógica fuzzy |

#### Operações Geométricas
| Operação | Descrição |
|----------|-----------|
| **Escala (Vizinho Mais Próximo)** | Redimensiona com interpolação pelo pixel mais próximo |
| **Escala (Bilinear)** | Redimensiona com interpolação bilinear (resultado mais suave) |
| **Rotação (Vizinho Mais Próximo)** | Rotaciona com interpolação pelo vizinho mais próximo. Aceita pivô customizado |
| **Rotação (Bilinear)** | Rotaciona com interpolação bilinear. Aceita pivô customizado |

> Para as rotações, deixe o Pivô X e Y como `0` para usar o centro da imagem automaticamente.

---

### 🌊 Transformada de Fourier

A página de Fourier tem seu **próprio upload de imagem** (independente do upload principal) e oferece três métodos de transformação.

#### Métodos Disponíveis

| Método | Complexidade | Observação |
|--------|-------------|------------|
| **DFT Manual** | O(N² × M²) — ingênua | Implementação completa sem bibliotecas. Imagens redimensionadas para máx. 64px por ser muito lenta |
| **FFT Manual** ⭐ | O(N·logN × M·logM) — Cooley-Tukey | Implementação manual do algoritmo recursivo. Imagens redimensionadas para potências de 2 |
| **FFT do NumPy** | Rápida | `np.fft.fft2` — alternativa rápida para imagens grandes, conforme exigido pelo enunciado |

> ⭐ A FFT Manual foi implementada inteiramente do zero (algoritmo de Cooley-Tukey), conforme a opção bônus do trabalho.

#### Como usar

1. Faça o upload da imagem na barra lateral da página de Fourier
2. Escolha o método de transformação
3. Clique em **"▶ Calcular Transformada"**
4. O espectro centralizado (escala logarítmica) aparecerá na tela
5. Use o **pincel** sobre o espectro para modificar frequências:
   - 🖤 **Preto** — zera a magnitude (remove frequências)
   - 🤍 **Branco** — maximiza a magnitude
   - 🔘 **Cinza suave** — define magnitude intermediária com slider de intensidade
6. Clique em **"✅ Aplicar edição e calcular inversa"** para ver a imagem filtrada
7. Use **"🔄 Restaurar"** para voltar ao espectro original
8. Baixe o resultado com o botão de download

> **Exemplo prático:** Risque com preto as linhas verticais do espectro → remove frequências que causam padrões horizontais repetitivos (filtro notch manual).

---

### 🎨 Processamento de Cores

| Operação | Descrição |
|----------|-----------|
| **Tons de Cinza (Média)** | `I = (R + G + B) / 3` |
| **Tons de Cinza (Ponderada NTSC)** | `I = 0.299R + 0.587G + 0.114B` (padrão televisivo) |
| **Chroma-Key** | Remove fundo verde substituindo-o por uma imagem de fundo. A detecção usa o espaço HSV com tolerância de matiz ajustável |
| **Equalização (HSI)** | Equaliza o histograma apenas no canal de Intensidade (I) do espaço HSI, preservando matiz e saturação |
| **Ajuste Matiz, Sat, Brilho** | Ajusta H (deslocamento angular), S (escala) e I (deslocamento) no espaço HSI |
| **Ajuste de Canais RGB** | Multiplica individualmente os canais R, G e B por fatores ajustáveis |
| **Sépia** | Aplica a transformação de sépia via matriz de cor clássica |

#### Conversão RGB ↔ HSI
O sistema HSI (Hue, Saturation, Intensity) é implementado manualmente em `modulos/cores.py`:
- `rgb_to_hsi(array)` → converte array RGB para HSI (H: 0–360°, S: 0–1, I: 0–255)
- `hsi_to_rgb(array)` → converte de volta para RGB com tratamento dos três setores angulares

---

## 🔧 Dependências Principais

| Pacote | Uso |
|--------|-----|
| `streamlit` | Interface web |
| `numpy` | Operações matriciais vetorizadas |
| `Pillow` | Abertura, conversão e salvamento de imagens |
| `matplotlib` | Plotagem de histogramas |
| `streamlit-drawable-canvas` | Canvas de desenho interativo (edição do espectro) 

---

## 📌 Observações Técnicas

- **Imagens coloridas:** Todas as operações de intensidade e filtragem espacial usam um decorador `@process_channels` que aplica automaticamente a função canal a canal (R, G, B) se a imagem for RGB.
- **Normalização de filtros:** Filtros como Sobel e Laplaciano podem gerar valores fora de [0, 255]. O checkbox **"Normalizar para exibição"** mapeia o resultado para [0, 255] linearmente, gerando a aparência "acinzentada" esperada.
- **FFT e potência de 2:** A FFT manual exige que as dimensões da imagem sejam potências de 2. A aplicação faz o redimensionamento automaticamente.
- **DFT ingênua:** É extremamente lenta para imagens grandes. A aplicação limita automaticamente a 64×64 pixels para evitar travamentos.
