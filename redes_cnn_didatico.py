"""
Como uma CNN aprende a enxergar: app didático em Streamlit (versão 3)

Como executar:
    pip install streamlit numpy opencv-python-headless matplotlib pandas
    streamlit run app_cnn_didatico3.py

O que este app mostra, em cada etapa:
  1. a explicação (com analogias entre parênteses, como se fosse para uma criança);
  2. controles para você mexer, sempre com o aviso do que aquilo muda no treinamento;
  3. os números calculados de verdade;
  4. o código-fonte REAL que está rodando, incluindo como ler a imagem com OpenCV.

Ideia central: NÃO EXISTE MÁGICA. Existe multiplicação, soma e ajustes pequenos repetidos muitas vezes.
"""
import inspect
import math

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================================
# FUNÇÕES DO PIPELINE (estas funções são executadas E exibidas como código)
# =====================================================================

def ler_imagem(bytes_arquivo):
    """Etapa 1: transforma os bytes do arquivo em uma tabela de números (RGB)."""
    dados = np.frombuffer(bytes_arquivo, dtype=np.uint8)
    imagem_bgr = cv2.imdecode(dados, cv2.IMREAD_COLOR)        # o OpenCV lê em BGR
    if imagem_bgr is None:                                     # arquivo inválido
        return None
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    return imagem_rgb


def isolar_canais(imagem_rgb, canais_ligados):
    """Etapa 1: mantém só os canais escolhidos (0=R, 1=G, 2=B) e zera os outros.
    Assim, o canal vermelho aparece VERMELHO, o verde aparece VERDE e o azul aparece AZUL."""
    resultado = np.zeros_like(imagem_rgb)
    for i in canais_ligados:
        resultado[:, :, i] = imagem_rgb[:, :, i]
    return resultado


def preprocessar(imagem_rgb, tamanho, cinza, normalizar):
    """Etapa 2: deixa a imagem no formato que a rede espera."""
    imagem = cv2.resize(imagem_rgb, (tamanho, tamanho))       # mesmo tamanho para todas
    if cinza:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)     # 3 canais -> 1 canal
    imagem = imagem.astype(np.float32)
    if normalizar:
        imagem = imagem / 255.0                               # valores entre 0 e 1
    return imagem


def convolucao(imagem, kernel, bias=0.0):
    """Etapa 3: desliza o kernel pela imagem. Cada posição vira UM número."""
    k = kernel.shape[0]
    altura, largura = imagem.shape
    saida = np.zeros((altura - k + 1, largura - k + 1), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            # soma ponderada: cada peso do kernel multiplica o pixel correspondente
            saida += kernel[i, j] * imagem[i:i + altura - k + 1, j:j + largura - k + 1]
    return saida + bias


def relu(mapa):
    """Etapa 4: valores negativos viram zero, positivos continuam iguais."""
    return np.maximum(0, mapa)


def pooling(mapa, tamanho, tipo):
    """Etapa 5: resume cada bloco (tamanho x tamanho) em um único número."""
    altura, largura = mapa.shape
    h2, w2 = altura // tamanho, largura // tamanho
    blocos = mapa[:h2 * tamanho, :w2 * tamanho].reshape(h2, tamanho, w2, tamanho)
    if tipo == "max":
        return blocos.max(axis=(1, 3))        # fica o maior valor do bloco
    return blocos.mean(axis=(1, 3))           # fica a média do bloco


def softmax(logits):
    """Etapa 6: transforma pontuações em probabilidades (somam 100%)."""
    e = np.exp(logits - logits.max())         # subtrai o maior para evitar overflow
    return e / e.sum()


def entropia_cruzada(probabilidades, classe_certa):
    """Etapa 6: o loss. Quanto menor a probabilidade da classe certa, maior o erro."""
    return float(-np.log(probabilidades[classe_certa] + 1e-12))


def gradiente_dos_logits(probabilidades, classe_certa):
    """Etapa 6: a 'direção de correção' de cada pontuação (resultado famoso: p - y)."""
    esperado = np.zeros_like(probabilidades)
    esperado[classe_certa] = 1.0              # a resposta certa vale 100%, as outras 0%
    return probabilidades - esperado


def padronizar(imagem):
    """Etapa 7: centraliza a imagem em zero (ajuda o treino a ser mais estável)."""
    return (imagem - imagem.mean()) / (imagem.std() + 1e-8)


def montar_amostras(imagem, k=3):
    """Etapa 7: cada janela k x k da imagem vira UMA amostra com k*k números."""
    altura, largura = imagem.shape
    colunas = []
    for i in range(k):
        for j in range(k):
            colunas.append(imagem[i:i + altura - k + 1, j:j + largura - k + 1].ravel())
    return np.stack(colunas, axis=1)          # formato: (número de janelas, k*k)


def erro_medio(X, alvo, pesos, indices):
    """Erro quadrático médio entre o palpite da rede e o resultado desejado."""
    palpite = X[indices] @ pesos
    return float(np.mean((palpite - alvo[indices]) ** 2))


def passo_na_mao(X, alvo, pesos, taxa, lote):
    """Etapa 7: UM ajuste de pesos, com todas as contas separadas para você ver."""
    palpite = X[lote] @ pesos                      # 1. palpite da rede
    erro = palpite - alvo[lote]                    # 2. quanto errou
    gradiente = 2 * X[lote].T @ erro / len(lote)   # 3. como corrigir cada peso
    passo = taxa * gradiente                       # 4. tamanho do passo
    novos_pesos = pesos - passo                    # 5. ajusta os pesos
    return palpite, erro, gradiente, passo, novos_pesos


def treinar(X, alvo, epocas, taxa, batch, semente):
    """Etapa 7: o treinamento. Repete épocas, e cada época percorre vários batches."""
    rng = np.random.default_rng(semente)
    ordem = rng.permutation(len(X))
    n_val = int(0.2 * len(X))
    idx_val, idx_treino = ordem[:n_val], ordem[n_val:]      # 80% treino, 20% validação

    pesos = rng.normal(0, 0.5, size=X.shape[1])              # pesos iniciais aleatórios
    hist = {"pesos": [pesos.copy()],
            "erro_treino": [erro_medio(X, alvo, pesos, idx_treino)],
            "erro_val": [erro_medio(X, alvo, pesos, idx_val)],
            "divergiu": False}

    with np.errstate(all="ignore"):
        for epoca in range(epocas):                           # UMA ÉPOCA
            embaralhado = rng.permutation(idx_treino)         # embaralha o treino
            for inicio in range(0, len(embaralhado), batch):  # UM BATCH
                lote = embaralhado[inicio:inicio + batch]
                palpite = X[lote] @ pesos                     # 1. palpite da rede
                erro = palpite - alvo[lote]                   # 2. quanto errou
                gradiente = 2 * X[lote].T @ erro / len(lote)  # 3. como corrigir cada peso
                pesos = pesos - taxa * gradiente              # 4. ajusta os pesos

            e_treino = erro_medio(X, alvo, pesos, idx_treino)
            if not np.isfinite(e_treino) or e_treino > 1e6:   # o treino "explodiu"
                hist["divergiu"] = True
                break
            hist["pesos"].append(pesos.copy())
            hist["erro_treino"].append(e_treino)
            hist["erro_val"].append(erro_medio(X, alvo, pesos, idx_val))

    hist["n_treino"], hist["n_val"] = len(idx_treino), len(idx_val)
    return hist


def descer_montanha(w_inicial, taxa, passos, w_ideal=3.0):
    """Etapa 8: gradiente descendente em UM único peso. Loss = (w - w_ideal)^2."""
    caminho = [w_inicial]
    with np.errstate(all="ignore"):
        for _ in range(passos):
            w = caminho[-1]
            gradiente = 2 * (w - w_ideal)         # a inclinação da montanha neste ponto
            caminho.append(w - taxa * gradiente)  # desce no sentido contrário da inclinação
    return np.array(caminho)


@st.cache_data(show_spinner=False)
def treinar_com_cache(X, alvo, epocas, taxa, batch, semente):
    return treinar(X, alvo, epocas, taxa, batch, semente)


# =====================================================================
# TEXTOS DE CÓDIGO (mostram como LER a imagem com OpenCV em todas as etapas)
# =====================================================================

LEITURA_CV2 = '''# ------------------------------------------------------------
# COMO LER UMA IMAGEM COM OPENCV (vale para todas as etapas)
# ------------------------------------------------------------
import cv2
import numpy as np

# Opção A: ler do DISCO, quando você tem o caminho do arquivo
imagem_bgr = cv2.imread("foto.jpg", cv2.IMREAD_COLOR)    # devolve None se não achar o arquivo
if imagem_bgr is None:
    raise FileNotFoundError("Não achei foto.jpg")
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)  # OpenCV usa BGR, convertemos para RGB

# Opção B: ler de BYTES, que é o que o upload do Streamlit entrega
dados = np.frombuffer(arquivo.getvalue(), dtype=np.uint8)
imagem_bgr = cv2.imdecode(dados, cv2.IMREAD_COLOR)
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

print(imagem_rgb.shape)   # (altura, largura, 3)  -> 3 canais: R, G, B
print(imagem_rgb.dtype)   # uint8                 -> inteiros de 0 a 255
'''


def snippet_preprocesso(tamanho):
    return f'''# ------------------------------------------------------------
# DEPOIS DE LER, a imagem é preparada (etapa 2) antes de entrar na rede
# ------------------------------------------------------------
imagem = cv2.resize(imagem_rgb, ({tamanho}, {tamanho}))    # tamanho único
imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)          # 1 canal (cinza)
img = imagem.astype(np.float32) / 255.0                    # valores de 0 a 1
'''


# =====================================================================
# AUXILIARES DE INTERFACE
# =====================================================================

FILTROS = {
    "Identidade (não muda nada)": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    "Média (borrar)": (np.ones((3, 3)) / 9).tolist(),
    "Bordas verticais (Sobel X)": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "Bordas horizontais (Sobel Y)": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "Nitidez (sharpen)": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "Contornos (Laplaciano)": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
}

NOMES_CANAIS = ["Vermelho (R)", "Verde (G)", "Azul (B)"]
CORES_TABELA = ["Reds", "Greens", "Blues"]


def imagem_exemplo():
    img = np.full((180, 180, 3), 225, np.uint8)
    cv2.rectangle(img, (20, 25), (95, 100), (210, 60, 50), -1)
    cv2.circle(img, (125, 115), 38, (40, 100, 200), -1)
    cv2.line(img, (10, 165), (170, 135), (30, 30, 30), 5)
    cv2.putText(img, "CNN", (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def mostrar_mapa(mapa, titulo, divergente=False, caixa=None, tamanho_fig=3.3):
    """Desenha uma imagem ou mapa de números. 'caixa' = (linha, coluna, lado)."""
    fig, ax = plt.subplots(figsize=(tamanho_fig, tamanho_fig))
    if mapa.ndim == 3 and mapa.dtype != np.uint8 and mapa.max() > 1:
        mapa = mapa.astype(np.uint8)                       # cor sem normalizar (0 a 255) em float
    if divergente:
        limite = max(float(np.abs(mapa).max()), 1e-6)
        ax.imshow(mapa, cmap="RdBu_r", vmin=-limite, vmax=limite, interpolation="nearest")
    elif mapa.ndim == 2:
        if mapa.dtype == np.uint8:                         # fixa a escala em 0..255
            ax.imshow(mapa, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            ax.imshow(mapa, cmap="gray", interpolation="nearest")
    else:
        ax.imshow(mapa, interpolation="nearest")
    if caixa is not None:
        linha, coluna, lado = caixa
        ax.add_patch(Rectangle((coluna - 0.5, linha - 0.5), lado, lado,
                               fill=False, edgecolor="red", linewidth=2))
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def tabela(matriz, casas=2, gradiente=True, cmap="Blues"):
    df = pd.DataFrame(np.asarray(matriz))
    estilo = df.style.format(f"{{:.{casas}f}}")
    if gradiente:
        estilo = estilo.background_gradient(cmap=cmap, axis=None)
    st.dataframe(estilo)


def mostrar_codigo(*funcoes, titulo="Ver o código-fonte desta etapa", pre_tamanho=None):
    """Mostra, nesta ordem: (1) como LER a imagem com OpenCV, (2) o pré-processo, (3) as funções reais."""
    with st.expander(titulo):
        st.caption("As funções abaixo são o código REAL que o app executa, lido do próprio arquivo. "
                   "O bloco do topo mostra como a imagem chega até elas.")
        partes = [LEITURA_CV2]
        if pre_tamanho is not None:
            partes.append(snippet_preprocesso(pre_tamanho))
        partes += [inspect.getsource(f) for f in funcoes]
        st.code("\n\n".join(partes), language="python")


def experimente(*itens):
    st.info("**Experimente:**\n\n" + "\n".join(f"- {i}" for i in itens))


def crianca(texto):
    """Explicação em parênteses, como se fosse para uma criança."""
    st.markdown(f"🧒 *({texto})*")


def impacto(texto):
    """Diz, para cada controle, o que ele muda no pipeline de treinamento."""
    st.markdown(f"> ⚙️ **O que isso muda no pipeline:** {texto}")


# =====================================================================
# PÁGINA
# =====================================================================

st.set_page_config(page_title="Como uma CNN aprende", layout="wide")
st.title("Como uma CNN aprende a enxergar")
st.write("Envie uma imagem e percorra, etapa por etapa, o que acontece dentro de uma rede neural "
         "convolucional. Mexa nos controles e veja os números mudarem.")
st.success("**Missão deste app:** no final, você vai enxergar que **não existe mágica**. "
           "Treinar um modelo é fazer **muita multiplicação e soma**, corrigindo pequenos erros várias vezes.")

# ---------------- barra lateral: aquisição da imagem ----------------
with st.sidebar:
    st.header("Imagem de entrada")
    arquivo = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg", "bmp", "webp"])
    st.caption("Sem upload, o app usa uma imagem de exemplo.")
    impacto("é a matéria-prima de tudo. Toda conta daqui para frente parte dos números desta imagem.")

if arquivo is not None:
    imagem_rgb = ler_imagem(arquivo.getvalue())
    if imagem_rgb is None:
        st.error("Não consegui ler esta imagem. Tente outro arquivo.")
        st.stop()
    maior_lado = max(imagem_rgb.shape[:2])
    if maior_lado > 600:                       # evita imagens enormes
        escala = 600 / maior_lado
        imagem_rgb = cv2.resize(imagem_rgb, None, fx=escala, fy=escala)
else:
    imagem_rgb = imagem_exemplo()

abas = st.tabs(["Visão geral", "1. Aquisição", "2. Pré-processamento", "3. Convolução",
                "4. ReLU", "5. Pooling", "6. Decisão e Loss", "7. Treinamento",
                "8. Sem mágica: tudo conectado"])

# ---------------- VISÃO GERAL ----------------
with abas[0]:
    st.header("Visão geral: o caminho da imagem até o aprendizado")
    st.markdown("""
Uma CNN faz dois movimentos, repetidos muitas vezes:

- **Para frente (forward):** a imagem passa pelas camadas e a rede dá um **palpite**.
- **Para trás (backward):** medimos o **erro** do palpite e **ajustamos os pesos** para errar menos.

```
 imagem -> pré-processamento -> CONVOLUÇÃO -> ReLU -> POOLING -> ... -> DECISÃO (softmax)
                                                                              |
   pesos ajustados <- OTIMIZADOR <- GRADIENTE <- LOSS (erro) <- compara com a resposta certa
```
    """)
    crianca("Pense em aprender a fazer bolo. Você segue uma receita (a rede faz o palpite), prova o bolo e percebe que "
            "ficou doce demais (o loss), e na próxima vez coloca um pouquinho menos de açúcar (o ajuste dos pesos). "
            "Depois de muitos bolos, a receita fica boa. Ninguém usou mágica, só provou e corrigiu.")

    st.subheader("O mapa dos conceitos: como tudo se encaixa")
    st.markdown("""
| Conceito | Analogia para uma criança | A matemática por trás |
|---|---|---|
| **Perceptron** (1 neurônio) | Um juiz que dá nota a cada pista e decide pelo total | `soma(entrada x peso) + bias`, depois uma ativação |
| **Rede convolucional** | Uma lupa de 3 x 3 que passeia pela foto, usando **a mesma** regra em todo lugar | Um perceptron repetido em cada posição, com pesos compartilhados |
| **Época** | Fazer a lição de casa **inteira** uma vez | Uma passada por todas as amostras de treino |
| **Batch** | Corrigir a lição de 64 em 64 exercícios, e não de um em um | Grupo de amostras usado para calcular UM ajuste |
| **Taxa de aprendizado** | O tamanho do passo ao descer uma ladeira no escuro | O número que multiplica o gradiente |
| **Loss** | A nota de "quanto eu errei" | Uma fórmula (erro quadrático, entropia cruzada...) |
| **Gradiente** | A setinha que aponta "por aqui o erro cresce" | A derivada do loss em relação a cada peso |
    """)
    st.warning("Na aba 7 treinamos de verdade uma camada convolucional com 9 pesos, usando a sua imagem. "
               "É o mesmo mecanismo das redes gigantes, só que pequeno o bastante para você enxergar cada número. "
               "Na aba 8 amarramos tudo: perceptron, convolução, épocas e taxa de aprendizado.")

# ---------------- ETAPA 1 ----------------
with abas[1]:
    st.header("Etapa 1: Aquisição e leitura da imagem")
    st.write("Para o computador, uma imagem **é uma tabela de números**. Cada ponto (pixel) vai de 0 (escuro) a 255 (claro). "
             "Uma imagem colorida tem **3 tabelas**, chamadas canais: vermelho (R), verde (G) e azul (B).")
    crianca("Imagine um mosaico de azulejos. Cada azulejo é um pixel. Para pintar a cor de cada azulejo, o computador usa "
            "três potinhos de tinta: um vermelho, um verde e um azul. O número diz quanta tinta de cada potinho vai no azulejo. "
            "0 é nenhuma tinta, 255 é o potinho todo.")

    altura, largura, canais = imagem_rgb.shape
    c1, c2 = st.columns([1, 1])
    with c1:
        sem_conversao = st.checkbox("Mostrar SEM converter de BGR para RGB (veja as cores trocarem)")
        impacto("não muda nada nas contas, só o que você vê. Mas, se você esquecesse de converter no código, a rede "
                "aprenderia com azul e vermelho trocados, e tudo que ela aprendesse ficaria enviesado.")
        mostrar_mapa(imagem_rgb[:, :, ::-1] if sem_conversao else imagem_rgb,
                     "Imagem lida" + (" (BGR mostrado como RGB)" if sem_conversao else ""))
        st.caption("O OpenCV lê as cores na ordem BGR. Se esquecermos de converter, azul e vermelho trocam de lugar. "
                   "(Como calçar o sapato direito no pé esquerdo: serve, mas fica tudo errado.)")
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Altura", f"{altura} px")
        m2.metric("Largura", f"{largura} px")
        m3.metric("Canais", canais)
        st.metric("Total de números na imagem", f"{altura * largura * canais:,}".replace(",", "."))
        st.write(f"Tipo dos valores: `{imagem_rgb.dtype}` (de {imagem_rgb.min()} até {imagem_rgb.max()})")
        st.caption("Conta: altura x largura x canais. É esse tanto de número que a rede precisa digerir.")

    st.subheader("Os três canais separados")
    st.write("Antes, os três canais apareciam todos em cinza, e ficava difícil perceber a diferença entre eles. "
             "Agora cada canal é mostrado **na própria cor**. Quanto mais forte a cor, mais tinta daquele potinho.")
    modo_canal = st.radio("Como pintar cada canal", ["Na cor do canal", "Em tons de cinza"], horizontal=True)
    impacto("só muda a pintura na tela. Os números dos canais são exatamente os mesmos nos dois modos. "
            "Em cinza, o branco significa 'muita tinta'. Na cor do canal, a cor forte significa 'muita tinta'.")
    cols = st.columns(3)
    for coluna, nome, indice in zip(cols, NOMES_CANAIS, range(3)):
        with coluna:
            if modo_canal == "Na cor do canal":
                mostrar_mapa(isolar_canais(imagem_rgb, [indice]), nome, tamanho_fig=2.6)
            else:
                mostrar_mapa(imagem_rgb[:, :, indice], nome + " (cinza)", tamanho_fig=2.6)
    crianca("Se a sua camiseta é vermelha, o potinho vermelho vem cheio nessa região, e os potinhos verde e azul quase vazios. "
            "Repare como o quadrado vermelho do exemplo brilha no canal R e fica escuro nos canais G e B.")

    st.subheader("Misture os canais e veja a cor nascer")
    ligados = st.multiselect("Potinhos de tinta ligados", NOMES_CANAIS, default=NOMES_CANAIS)
    impacto("é só uma demonstração de que a imagem colorida é a **soma** dos 3 canais. Para a rede, desligar um canal "
            "equivale a esconder uma parte da informação que ela poderia usar para decidir.")
    mix = isolar_canais(imagem_rgb, [NOMES_CANAIS.index(n) for n in ligados])
    mostrar_mapa(mix, "Imagem com os canais escolhidos", tamanho_fig=3.0)

    st.subheader("Os números de verdade")
    st.write("Escolha um pedaço 8 x 8 da imagem e um canal para ver os valores dos pixels.")
    k1, k2, k3 = st.columns(3)
    canal_nome = k1.selectbox("Canal", NOMES_CANAIS)
    lin0 = k2.slider("Linha inicial", 0, max(altura - 8, 0), min(altura // 3, max(altura - 8, 0)))
    col0 = k3.slider("Coluna inicial", 0, max(largura - 8, 0), min(largura // 3, max(largura - 8, 0)))
    impacto("nada é treinado aqui. Você está só espiando os números que a rede vai receber como entrada. "
            "Mas é importante enxergar: **a rede nunca vê a foto, ela vê só esta tabela de números**.")
    canal_idx = NOMES_CANAIS.index(canal_nome)
    tabela(imagem_rgb[lin0:lin0 + 8, col0:col0 + 8, canal_idx], casas=0, cmap=CORES_TABELA[canal_idx])
    mostrar_codigo(ler_imagem, isolar_canais)
    experimente("Marque a caixa de BGR e compare as cores.",
                "Mude para 'Em tons de cinza' e perceba que os três canais parecem iguais em cor, mas têm números diferentes.",
                "Mude a linha e a coluna inicial para passar por uma borda e veja os números pularem de valor.")

# ---------------- ETAPA 2 ----------------
with abas[2]:
    st.header("Etapa 2: Pré-processamento")
    st.write("A rede só aceita entradas **padronizadas**. Precisamos de um tamanho único, normalmente 1 ou 3 canais, "
             "e valores pequenos (entre 0 e 1) para o treino ser estável.")
    crianca("É como arrumar os ingredientes antes de cozinhar: todos picados no mesmo tamanho e medidos na mesma colher. "
            "Se um vem inteiro e o outro em pedacinhos, a receita não funciona direito.")
    p1, p2, p3 = st.columns(3)
    tamanho = p1.slider("Tamanho da imagem (pixels)", 16, 128, 32, step=8)
    with p1:
        impacto("define **quantas entradas** a rede tem (tamanho x tamanho). Maior = mais detalhe, mas "
                "o número de contas cresce com o **quadrado** do tamanho (dobrou o lado, 4 vezes mais contas por época).")
    cinza = p2.checkbox("Converter para cinza (1 canal)", value=True)
    with p2:
        impacto("de 3 canais para 1 reduz as entradas e as contas em **3 vezes**. Em troca, perde-se a informação de cor "
                "(uma maçã vermelha e uma verde ficam parecidas).")
    normalizar = p3.checkbox("Normalizar (dividir por 255)", value=True)
    with p3:
        impacto("deixa os números entre 0 e 1. Sem isso, os valores grandes (até 255) geram gradientes enormes, "
                "e a taxa de aprendizado precisaria ser minúscula para o treino não explodir.")

    processada = preprocessar(imagem_rgb, tamanho, cinza, normalizar)
    c1, c2 = st.columns(2)
    with c1:
        mostrar_mapa(imagem_rgb, f"Antes: {imagem_rgb.shape}")
    with c2:
        mostrar_mapa(processada, f"Depois: {processada.shape}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Menor valor", f"{processada.min():.2f}")
    m2.metric("Maior valor", f"{processada.max():.2f}")
    m3.metric("Números na entrada da rede", f"{processada.size:,}".replace(",", "."))
    st.caption(f"Conta: {tamanho} x {tamanho} x {1 if cinza else 3} canais = {processada.size} números. "
               f"Uma única convolução 3 x 3 faria cerca de {(tamanho - 2) ** 2 * 9 * (1 if cinza else 3):,} multiplicações nesta imagem."
               .replace(",", "."))
    st.write("Primeiros 8 x 8 valores da imagem processada" + (" (canal R)" if not cinza else "") + ":")
    bloco = processada[:8, :8] if processada.ndim == 2 else processada[:8, :8, 0]
    tabela(bloco, casas=2 if normalizar else 0)
    mostrar_codigo(preprocessar)
    st.caption("Daqui em diante usamos a imagem em **cinza e normalizada**, com o tamanho escolhido acima, para simplificar as contas.")
    experimente("Diminua o tamanho para 16 e veja quantos detalhes se perdem.",
                "Desmarque 'Normalizar' e veja os valores voltarem para a faixa de 0 a 255.",
                "Desmarque 'cinza' e veja o número de entradas triplicar.")

# imagem usada nas próximas etapas
img = preprocessar(imagem_rgb, tamanho, True, True)

# ---------------- ETAPA 3 ----------------
with abas[3]:
    st.header("Etapa 3: Convolução")
    st.write("O **kernel** (ou filtro) é uma pequena tabela de **pesos** 3 x 3. Ele desliza pela imagem. "
             "Em cada posição, multiplicamos o kernel pelos pixels que ele cobre e **somamos tudo**: o resultado é "
             "um único pixel do **mapa de características**. Kernels diferentes detectam coisas diferentes (bordas, texturas...).")
    crianca("Imagine uma lupa quadradinha com 9 buraquinhos, cada um com uma 'força' (o peso). Você passeia com a lupa pela foto "
            "e, em cada lugar, soma o que vê pelos buraquinhos, cada um com a sua força. Se a lupa foi feita para achar riscos "
            "de pé, ela 'grita' (número grande) quando passa por um risco de pé.")
    preset = st.selectbox("Escolha um filtro pronto (depois você pode editar os números)",
                          list(FILTROS.keys()), index=2)
    impacto("o filtro é **o que a rede aprende**. Na etapa 7, estes 9 números começam aleatórios e o treino os ajusta "
            "até virarem um detector útil. Escolher um filtro pronto equivale a 'colar a resposta'.")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.write("**Kernel (edite os pesos!)**")
        df_k = pd.DataFrame(FILTROS[preset], columns=["c1", "c2", "c3"], index=["l1", "l2", "l3"]).astype(float)
        editado = st.data_editor(df_k, key=f"kernel_{preset}")
        kernel = np.nan_to_num(editado.to_numpy(dtype=np.float32))
        impacto("cada peso decide **quanto aquele pixel vizinho importa**. Peso positivo soma, negativo subtrai, zero ignora. "
                "Mudar um peso muda o mapa inteiro, e é exatamente isso que o treino faz, só que automaticamente.")
    mapa = convolucao(img, kernel)
    with c2:
        mostrar_mapa(img, "Entrada (cinza)")
    with c3:
        mostrar_mapa(mapa, "Mapa de características", divergente=True)
        st.caption("Vermelho = valores positivos, azul = negativos, branco = perto de zero.")

    h, w = img.shape
    st.markdown(f"**Tamanho da saída:** `(entrada - kernel) + 1` = ({h} - 3) + 1 = **{h - 2}**. "
                f"A saída fica menor porque o kernel não cabe nas bordas. "
                f"(Se a lupa passa da beirada da foto, ela não enxerga o que está fora.)")
    n_contas = (h - 2) * (w - 2) * 9
    st.caption(f"Conta total desta etapa: {(h - 2)} x {(w - 2)} posições x 9 multiplicações = **{n_contas:,}** multiplicações, "
               f"mais as somas. Parâmetros treináveis: apenas **9 pesos** (+1 bias).".replace(",", "."))

    st.subheader("Uma única conta, passo a passo")
    s1, s2 = st.columns(2)
    linha = s1.slider("Linha da posição", 0, h - 3, min(h // 2, h - 3), key="lin_conv")
    coluna = s2.slider("Coluna da posição", 0, w - 3, min(w // 2, w - 3), key="col_conv")
    impacto("só escolhe **onde** olhar. Não treina nada, mas mostra que o kernel **usa os mesmos 9 pesos em todas as posições** "
            "(isso se chama compartilhamento de pesos, e é o que torna a CNN tão econômica).")
    recorte = img[linha:linha + 3, coluna:coluna + 3]
    produto = recorte * kernel
    d1, d2, d3, d4 = st.columns([1.2, 1, 1, 1])
    with d1:
        mostrar_mapa(img, "Onde o kernel está (caixa vermelha)", caixa=(linha, coluna, 3))
    with d2:
        st.write("Pixels cobertos")
        tabela(recorte, 2)
    with d3:
        st.write("Kernel")
        tabela(kernel, 2, gradiente=False)
    with d4:
        st.write("Pixel x peso")
        tabela(produto, 2, gradiente=False)
    st.success(f"Soma de todos os produtos = **{produto.sum():.3f}**   (o mapa de características nessa posição vale {mapa[linha, coluna]:.3f})")
    mostrar_codigo(convolucao, pre_tamanho=tamanho)
    experimente("Escolha 'Bordas verticais' e arraste a coluna até cruzar a borda do quadrado: a soma fica grande.",
                "Em uma região lisa, a soma fica perto de zero. Por quê? (Todos os vizinhos são iguais, e os pesos positivos e negativos se cancelam.)",
                "Edite os pesos e crie o seu próprio filtro. Se todos forem 1/9, você recria o filtro da média.")

# ---------------- ETAPA 4 ----------------
with abas[4]:
    st.header("Etapa 4: ReLU (função de ativação)")
    st.write("A convolução só faz contas de soma e multiplicação. Sem algo extra, empilhar camadas seria o mesmo que uma única camada. "
             "A **ReLU** quebra essa linearidade com uma regra simples: **negativo vira zero, positivo continua igual**. "
             "O **bias** é um número somado antes da ReLU que desloca o 'ponto de corte'.")
    crianca("A ReLU é um porteiro de balada: quem chega com ponto positivo entra com o mesmo ponto, e quem chega com ponto "
            "negativo fica do lado de fora (vira zero). O bias é o quanto o porteiro está exigente hoje: com bias alto ele "
            "deixa quase todo mundo entrar, com bias baixo só os mais fortes passam.")
    bias = st.slider("Bias (somado ao mapa antes da ReLU)", -2.0, 2.0, 0.0, 0.05)
    impacto("o bias é **o 10º parâmetro treinável** do kernel. Ele decide o quão fácil é 'acender' o neurônio. "
            "Com bias muito baixo, quase tudo vira zero e o gradiente some (neurônio 'dormindo'). Durante o treino, "
            "a rede aprende o melhor valor sozinha.")
    antes = convolucao(img, kernel, bias)
    depois = relu(antes)
    c1, c2, c3 = st.columns(3)
    with c1:
        mostrar_mapa(antes, "Antes da ReLU", divergente=True)
    with c2:
        mostrar_mapa(depois, "Depois da ReLU")
    with c3:
        xs = np.linspace(-3, 3, 61)
        fig, ax = plt.subplots(figsize=(3.3, 3.3))
        ax.plot(xs, relu(xs))
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_title("Função ReLU", fontsize=10)
        st.pyplot(fig); plt.close(fig)
    zerados = float((antes <= 0).mean() * 100)
    st.metric("Valores que viraram zero", f"{zerados:.1f}%")
    st.caption("Onde o valor virou zero, o gradiente também vira zero: aquela posição não ajuda a corrigir os pesos naquele passo. "
               "(Quem ficou do lado de fora da balada não pode dar opinião sobre a festa.)")
    st.write("Comparando os primeiros 6 x 6 valores:")
    t1, t2 = st.columns(2)
    with t1:
        st.write("Antes"); tabela(antes[:6, :6], 2, gradiente=False)
    with t2:
        st.write("Depois"); tabela(depois[:6, :6], 2, gradiente=False)
    mostrar_codigo(relu, pre_tamanho=tamanho)
    experimente("Aumente o bias e veja mais da imagem 'acender'. Diminua e veja só os pontos mais fortes sobreviverem.")

# ---------------- ETAPA 5 ----------------
with abas[5]:
    st.header("Etapa 5: Pooling")
    st.write("O **pooling** resume cada bloco da imagem em um único número. Isso **diminui o tamanho**, deixa o cálculo mais rápido "
             "e faz a rede tolerar pequenos deslocamentos do objeto. O **max pooling** guarda o maior valor do bloco.")
    crianca("É como fazer o resumo de um capítulo: em vez de decorar o texto todo, você guarda só a frase mais importante "
            "de cada parágrafo. No max pooling, a 'frase importante' é o número mais forte do bloco.")
    q1, q2 = st.columns(2)
    tipo = q1.radio("Tipo de pooling", ["max", "média"], horizontal=True)
    with q1:
        impacto("**não tem pesos para treinar**, é só uma regra fixa. O max destaca o sinal mais forte (bom para achar bordas), "
                "a média suaviza e guarda um resumo geral.")
    tam_pool = q2.slider("Tamanho do bloco", 2, 4, 2)
    with q2:
        impacto("quanto maior o bloco, menos números seguem para a próxima camada, e menos pesos ela precisa (treino mais rápido). "
                "O preço é perder detalhe e posição exata.")
    tipo_func = "max" if tipo == "max" else "media"
    reduzido = pooling(depois, tam_pool, tipo_func)
    c1, c2 = st.columns(2)
    with c1:
        mostrar_mapa(depois, f"Antes: {depois.shape}")
    with c2:
        mostrar_mapa(reduzido, f"Depois ({tipo}): {reduzido.shape}")
    st.metric("Redução de números", f"{depois.size} -> {reduzido.size}",
              delta=f"{(1 - reduzido.size / depois.size) * 100:.0f}% menos", delta_color="off")

    st.subheader("Um bloco, passo a passo")
    b1, b2 = st.columns(2)
    bl = b1.slider("Linha do bloco", 0, reduzido.shape[0] - 1, reduzido.shape[0] // 2)
    bc = b2.slider("Coluna do bloco", 0, reduzido.shape[1] - 1, reduzido.shape[1] // 2)
    impacto("só escolhe qual bloco inspecionar. Não altera o treino.")
    bloco = depois[bl * tam_pool:(bl + 1) * tam_pool, bc * tam_pool:(bc + 1) * tam_pool]
    e1, e2 = st.columns(2)
    with e1:
        st.write("Valores do bloco"); tabela(bloco, 2)
    with e2:
        st.metric(f"Resultado ({tipo})", f"{reduzido[bl, bc]:.3f}")
    mostrar_codigo(pooling, pre_tamanho=tamanho)
    st.write("**Flatten:** no final das camadas convolucionais, o mapa é esticado em uma fila de números "
             "que alimenta a camada de decisão. (Como pegar uma folha quadriculada e cortar as linhas para "
             "colar uma atrás da outra, formando uma fita comprida.)")
    st.write(f"Aqui, o mapa de {reduzido.shape[0]} x {reduzido.shape[1]} vira um vetor com **{reduzido.size} números**.")
    st.caption(f"Impacto prático: se a decisão final tem 3 classes, a camada final precisa de {reduzido.size} x 3 = "
               f"**{reduzido.size * 3} pesos** para treinar. Com pooling maior, esse número cai.")
    experimente("Troque max por média e compare as bordas.", "Aumente o bloco para 4 e veja quanto detalhe se perde.")

# ---------------- ETAPA 6 ----------------
with abas[6]:
    st.header("Etapa 6: Decisão (softmax) e Loss")
    st.write("A última camada da rede entrega uma **pontuação (logit)** para cada classe. O **softmax** transforma essas pontuações "
             "em **probabilidades** que somam 100%. Depois o **loss** mede o erro: se a rede dá pouca probabilidade para a "
             "classe correta, o loss é alto.")
    crianca("Pense num programa de talentos. Cada jurado dá uma nota (logit) a cada candidato. O softmax transforma as notas "
            "em 'porcentagem de torcida' para que a soma dê 100%. O loss é o susto que a gente leva quando o candidato que "
            "era o vencedor de verdade recebe pouca torcida.")
    nomes_txt = st.text_input("Nomes das classes (separados por vírgula)", "gato, cachorro, pássaro")
    impacto("define quantas saídas a rede tem. Mais classes significa mais pesos na camada final e um problema mais difícil.")
    classes = [n.strip() for n in nomes_txt.split(",") if n.strip()]
    if len(classes) < 2:
        st.warning("Informe pelo menos 2 classes.")
    else:
        certa = st.selectbox("Qual é a classe CORRETA desta imagem?", classes)
        impacto("é o **gabarito**. Sem gabarito não existe loss, e sem loss a rede não sabe para onde se corrigir. "
                "Isso é o que chamamos de aprendizado supervisionado.")
        idx_certa = classes.index(certa)
        st.write("**Mexa nas pontuações (logits) que a rede deu para cada classe:**")
        logits = np.array([st.slider(f"Pontuação de '{c}'", -6.0, 6.0, float(2 - i), 0.1, key=f"logit_{i}_{c}")
                           for i, c in enumerate(classes)])
        impacto("estas pontuações são o **resultado das contas dos pesos** (convolução, ReLU, pooling, camada final). "
                "No treino real você não mexe nelas: quem as move são os pesos, ajustados pelo gradiente.")
        probs = softmax(logits)
        loss = entropia_cruzada(probs, idx_certa)
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(pd.DataFrame({"Probabilidade": probs}, index=classes))
        with c2:
            st.dataframe(pd.DataFrame({"Pontuação": logits.round(2),
                                       "Probabilidade": [f"{p:.1%}" for p in probs]}, index=classes))
            st.metric("Loss (entropia cruzada)", f"{loss:.3f}")
            palpite = classes[int(probs.argmax())]
            (st.success if palpite == certa else st.error)(f"Palpite da rede: {palpite}")
        st.latex(r"\text{loss} = -\ln(\text{probabilidade da classe correta})")
        st.write(f"Neste caso: -ln({probs[idx_certa]:.3f}) = **{loss:.3f}**")

        st.subheader("De onde vem a correção: o gradiente")
        grad_l = gradiente_dos_logits(probs, idx_certa)
        st.write("Para cada pontuação, o gradiente diz **para que lado e com que força** ela deveria se mexer. "
                 "Aqui a conta é lindamente simples: `probabilidade - resposta esperada`.")
        st.dataframe(pd.DataFrame({"Probabilidade": probs.round(3),
                                   "Esperado": [1.0 if i == idx_certa else 0.0 for i in range(len(classes))],
                                   "Gradiente (prob - esperado)": grad_l.round(3)}, index=classes))
        crianca("Se a classe certa tem gradiente negativo, o recado é 'suba essa nota!'. Se uma classe errada tem gradiente "
                "positivo, o recado é 'desça essa nota!'. Quanto maior o erro, mais forte o recado.")
        mostrar_codigo(softmax, entropia_cruzada, gradiente_dos_logits, pre_tamanho=tamanho)
        experimente("Aumente a pontuação da classe correta e veja o loss cair.",
                    "Aumente a pontuação de uma classe ERRADA e veja o loss disparar.",
                    "Repare no gradiente: quando a rede acerta com folga, ele fica perto de zero e quase não há o que corrigir.",
                    "Treinar é empurrar o loss para perto de zero, ajustando os pesos.")

# ---------------- ETAPA 7 ----------------
with abas[7]:
    st.header("Etapa 7: Treinamento (a rede aprendendo)")
    st.write("""
Vamos treinar de verdade **um kernel 3 x 3** (9 pesos) com a sua imagem. A tarefa: **descobrir sozinho** qual filtro
produz o resultado desejado. Você escolhe um filtro-alvo, a rede começa com **pesos aleatórios** e vai se corrigindo.

Os conceitos aparecem aqui ao vivo:
- **Amostra:** cada janela 3 x 3 da imagem. **Batch:** grupo de amostras usado para calcular um ajuste.
- **Época:** uma passada por **todas** as amostras de treino (vários batches).
- **Loss:** erro quadrático médio entre o palpite e o resultado desejado.
- **Taxa de aprendizado:** o tamanho do passo de cada ajuste.
- **Validação:** 20% das amostras ficam de fora do treino, só para testar.
    """)
    crianca("Imagine que você está de olhos vendados, tentando achar o ponto mais baixo de um vale. A cada passo você tateia o chão, "
            "sente para que lado ele desce e dá um passinho nessa direção. A taxa de aprendizado é o tamanho do passo, "
            "a época é uma volta completa pelo terreno e o batch é quantos pontinhos do chão você tateia antes de dar cada passo.")

    cA, cB, cC = st.columns(3)
    alvo_nome = cA.selectbox("Filtro-alvo (o que a rede deve aprender)",
                             [f for f in FILTROS if not f.startswith("Identidade")], index=1, key="alvo")
    with cA:
        impacto("é a **resposta certa** do treino. Define o 'gabarito' e, com ele, o tamanho do erro inicial. "
                "Filtros com números grandes (como o Laplaciano) geram erros maiores no começo.")
    epocas = cB.slider("Épocas", 1, 60, 20)
    with cB:
        impacto("quantas **voltas completas** pelos dados. Mais épocas = mais ajustes = mais aprendizado, até o erro parar de cair. "
                "Cada época a mais custa tempo de conta.")
    taxa = cC.select_slider("Taxa de aprendizado", [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0], value=0.05)
    with cC:
        impacto("o **tamanho de cada passo**. Muito pequena: aprende devagar e precisa de muitas épocas. "
                "Muito grande: passa direto do ponto ideal e o erro **explode**.")
    cD, cE = st.columns(2)
    batch = cD.select_slider("Tamanho do batch", [4, 16, 64, 256, 1024, 4096], value=64)
    with cD:
        impacto("quantas amostras entram em **cada ajuste**. Batch pequeno: muitos ajustes por época, porém 'barulhentos'. "
                "Batch grande: poucos ajustes por época, porém mais suaves e estáveis.")
    semente = cE.number_input("Semente (muda os pesos iniciais)", 0, 999, 0)
    with cE:
        impacto("escolhe de onde a rede **começa** (pesos aleatórios). Sementes diferentes começam em pontos diferentes, "
                "mas um bom treino chega ao mesmo destino.")

    img_z = padronizar(img)
    X = montar_amostras(img_z)
    kernel_alvo = np.array(FILTROS[alvo_nome], dtype=np.float32)
    alvo = convolucao(img_z, kernel_alvo).ravel()

    with st.spinner("Treinando..."):
        hist = treinar_com_cache(X, alvo, epocas, float(taxa), int(batch), int(semente))

    n_tr = hist["n_treino"]
    lotes = math.ceil(n_tr / batch)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Amostras de treino", n_tr)
    m2.metric("Amostras de validação", hist["n_val"])
    m3.metric("Batches por época", lotes)
    m4.metric("Ajustes de peso por época", lotes)
    st.caption(f"Conta: {n_tr} amostras / batch de {batch} = {lotes} batches por época. "
               f"Em {epocas} épocas são {lotes * epocas} ajustes de peso no total.")

    epocas_feitas = len(hist["pesos"]) - 1
    if hist["divergiu"]:
        st.error(f"O treino DIVERGIU na época {epocas_feitas + 1}: o erro explodiu. "
                 "Os passos foram grandes demais. Diminua a taxa de aprendizado. "
                 "(Era um passo tão grande que você pulou o vale inteiro e caiu na montanha do outro lado.)")
    elif hist["erro_treino"][-1] < 0.01 * hist["erro_treino"][0]:
        st.success("Diagnóstico: a rede **aprendeu bem**. O erro final é menos de 1% do erro inicial.")
    else:
        st.warning("Diagnóstico: a rede ainda está aprendendo. Tente **mais épocas** ou uma **taxa um pouco maior** "
                   "(sem exagerar, para não divergir).")

    st.subheader("O erro (loss) ao longo das épocas")
    escala_log = st.checkbox("Escala logarítmica (ajuda a ver erros pequenos)", value=True)
    impacto("só muda o jeito de desenhar o gráfico. Na escala log, cada degrau vertical é uma divisão por 10 do erro.")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = list(range(len(hist["erro_treino"])))
    ax.plot(xs, hist["erro_treino"], label="Treino")
    ax.plot(xs, hist["erro_val"], label="Validação", linestyle="--")
    if escala_log:
        ax.set_yscale("log")
    ax.set_xlabel("Época (0 = antes de treinar)")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig); plt.close(fig)

    st.subheader("O que a rede aprendeu em cada época")
    epoca_ver = st.slider("Ver o estado da rede na época", 0, max(epocas_feitas, 1), epocas_feitas)
    impacto("é uma 'máquina do tempo': mostra o kernel exatamente como estava naquela época. Não altera o treino.")
    epoca_ver = min(epoca_ver, epocas_feitas)
    pesos_ep = hist["pesos"][epoca_ver].reshape(3, 3)
    saida_ep = convolucao(img_z, pesos_ep.astype(np.float32))
    alvo_img = alvo.reshape(saida_ep.shape)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.write("**Kernel aprendido**"); tabela(pesos_ep, 3)
    with g2:
        st.write("**Kernel-alvo (resposta certa)**"); tabela(kernel_alvo, 3, gradiente=False)
    with g3:
        st.write("**Diferença**"); tabela(pesos_ep - kernel_alvo, 3, gradiente=False)
    i1, i2, i3 = st.columns(3)
    with i1:
        mostrar_mapa(saida_ep, f"Saída da rede (época {epoca_ver})", divergente=True)
    with i2:
        mostrar_mapa(alvo_img, "Saída desejada", divergente=True)
    with i3:
        mostrar_mapa(saida_ep - alvo_img, "Erro (saída - desejada)", divergente=True)

    st.subheader("Como cada um dos 9 pesos mudou")
    evolucao = pd.DataFrame(np.array(hist["pesos"]), columns=[f"peso {i + 1}" for i in range(9)])
    evolucao.index.name = "época"
    st.line_chart(evolucao)

    st.subheader("A regra de ajuste (o coração do treinamento)")
    st.latex(r"\text{novo peso} = \text{peso} - \text{taxa} \times \text{gradiente}")
    st.write("O **gradiente** diz, para cada peso, em que direção o erro cresce. Andamos no sentido contrário, "
             "com um passo de tamanho `taxa`. Como o alpinista descendo a montanha no escuro.")

    st.subheader("Um único ajuste, com todas as contas na mesa")
    st.write("Abaixo, pegamos **4 amostras** (um mini-batch) e os **pesos iniciais** da rede, e fazemos **um** ajuste na mão. "
             "É isso, e só isso, que se repete milhares de vezes no treino.")
    lote_demo = np.random.default_rng(int(semente)).choice(len(X), 4, replace=False)
    pesos0 = hist["pesos"][0]
    palpite_d, erro_d, grad_d, passo_d, novos_d = passo_na_mao(X, alvo, pesos0, float(taxa), lote_demo)
    st.markdown("**Passo 1 e 2: o palpite e o erro de cada amostra**")
    st.dataframe(pd.DataFrame({"Palpite da rede": palpite_d.round(3),
                               "Resposta desejada": alvo[lote_demo].round(3),
                               "Erro (palpite - desejado)": erro_d.round(3),
                               "Erro ao quadrado": (erro_d ** 2).round(3)}))
    st.markdown("**Passo 3, 4 e 5: o gradiente, o passo e o novo peso de cada um dos 9 pesos**")
    st.dataframe(pd.DataFrame({"Peso atual": pesos0.round(4),
                               "Gradiente": grad_d.round(4),
                               f"Passo (taxa {taxa} x gradiente)": passo_d.round(4),
                               "Novo peso": novos_d.round(4),
                               "Peso ideal (alvo)": kernel_alvo.ravel().round(4)},
                              index=[f"peso {i + 1}" for i in range(9)]))
    crianca("Cada linha da tabela é um botãozinho de volume. O gradiente diz 'gire para a esquerda' ou 'gire para a direita', "
            "a taxa diz 'gire só um tantinho assim', e o novo peso é o botão depois do giro. A rede tem 9 botões, "
            "e o treino é girar os 9 botões um tantinho, de novo e de novo.")
    mostrar_codigo(padronizar, montar_amostras, erro_medio, passo_na_mao, treinar,
                   titulo="Ver o código-fonte do treinamento", pre_tamanho=tamanho)
    experimente("Taxa 1.0: o treino diverge (passos grandes demais). Taxa 0.0001: aprende, mas muito devagar.",
                "Batch 4096 (quase todas as amostras) gera poucos ajustes por época. Batch 4 gera muitos, mas mais 'barulhentos'.",
                "Com 1 época a rede mal aprende. Com 40, o kernel aprendido fica quase igual ao alvo.",
                "Mude a semente: pesos iniciais diferentes, mas o destino é o mesmo.",
                "Compare a curva de treino com a de validação: quando ficam juntas, a rede aprendeu o padrão (sem overfitting).")

# ---------------- ETAPA 8 ----------------
with abas[8]:
    st.header("Etapa 8: Sem mágica, tudo conectado")
    st.write("Chegou a hora de amarrar as pontas. Vamos ligar **perceptron**, **rede convolucional**, **épocas** e "
             "**taxa de aprendizado** em uma única história.")
    st.markdown("""
```
 PERCEPTRON            REDE CONVOLUCIONAL            TREINAMENTO
 1 neurônio     ->     muitos neurônios,      ->     repete: palpite, erro, ajuste
 soma ponderada        mesmos pesos em todo          (cada volta completa = 1 ÉPOCA,
 + bias + ativação     lugar da imagem               cada ajuste = passo de tamanho TAXA)
```
    """)

    # ---------- 8.1 perceptron ----------
    st.subheader("8.1 O perceptron: o tijolinho de tudo")
    st.write("Um **perceptron** faz quatro coisas: recebe números, multiplica cada um por um **peso**, soma tudo, "
             "soma o **bias** e passa por uma **ativação** (aqui, a ReLU).")
    st.latex(r"\text{saída} = \text{ReLU}\left(x_1 w_1 + x_2 w_2 + \dots + x_9 w_9 + \text{bias}\right)")
    crianca("O perceptron é um juiz de concurso de bolo com 9 pistas: cor, cheiro, altura... Cada pista tem uma importância (o peso). "
            "Ele soma as notas ponderadas, dá um empurrãozinho (o bias) e só levanta a plaquinha se o total for positivo (a ReLU).")
    st.write("**O ponto surpreendente:** cada posição do kernel que você viu na etapa 3 **é exatamente um perceptron** com 9 entradas. "
             "Veja com os números reais da sua imagem, no centro dela, com o kernel da etapa 3 e o bias da etapa 4:")
    lin_c, col_c = (h - 2) // 2, (w - 2) // 2
    entradas = img[lin_c:lin_c + 3, col_c:col_c + 3].ravel()
    pesos_k = kernel.ravel()
    soma = float(entradas @ pesos_k)
    z = soma + bias
    saida_p = max(0.0, z)
    st.dataframe(pd.DataFrame({"Entrada x": entradas.round(3), "Peso w (do kernel)": pesos_k.round(3),
                               "x vezes w": (entradas * pesos_k).round(3)},
                              index=[f"x{i + 1}" for i in range(9)]))
    st.success(f"Soma dos produtos = {soma:.3f}   |   mais o bias ({bias:.2f}) = {z:.3f}   |   "
               f"depois da ReLU = **{saida_p:.3f}**")
    impacto("o kernel da aba 3 e o bias da aba 4 que você escolheu são usados aqui. Mudar qualquer um deles muda esta conta, "
            "e é exatamente por isso que treinar é mexer nesses números.")

    # ---------- 8.2 CNN: pesos compartilhados ----------
    st.subheader("8.2 Da rede densa para a convolucional: o truque dos pesos compartilhados")
    pesos_densa = (h * w) * ((h - 2) * (w - 2))
    pesos_conv = 9 + 1
    st.write("Se cada ponto da saída fosse ligado a **todos** os pixels da entrada (uma rede 'densa'), "
             "cada ligação teria o seu próprio peso. A CNN faz diferente: **um único perceptron de 9 entradas desliza pela imagem toda**.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Pesos na rede densa", f"{pesos_densa:,}".replace(",", "."))
    k2.metric("Pesos na convolução", pesos_conv)
    k3.metric("Economia", f"{pesos_densa // pesos_conv:,}x".replace(",", "."))
    crianca("Em vez de contratar um fiscal diferente para cada cantinho da escola, você treina UM fiscal esperto e ele "
            "passeia pela escola inteira. Ele aprende a reconhecer 'uma bagunça' em qualquer lugar, e você só pagou um salário.")
    impacto("menos pesos significa menos contas por época, menos memória e um treino que funciona até com poucos dados. "
            "A **imagem de entrada** (aba 2, tamanho) cresce o número de pesos da rede densa muito rápido, mas quase não muda o da convolução.")

    # ---------- 8.3 épocas + taxa ----------
    st.subheader("8.3 Épocas e taxa de aprendizado: a descida da montanha")
    st.write("Vamos treinar **um único peso** para ficar no valor ideal `w = 3`. O erro é `loss = (w - 3)²`, "
             "uma montanha em forma de tigela. O gradiente é a inclinação da tigela naquele ponto.")
    st.latex(r"\text{gradiente} = 2 \times (w - 3) \qquad\qquad w_{\text{novo}} = w - \text{taxa} \times \text{gradiente}")
    d1, d2, d3 = st.columns(3)
    w0 = d1.slider("Onde o peso começa (w inicial)", -8.0, 14.0, -4.0, 0.5, key="w0_montanha")
    with d1:
        impacto("é o equivalente da **semente** no treino real: de onde a rede começa. Mais longe do ideal, mais passos para chegar.")
    taxa_m = d2.slider("Taxa de aprendizado", 0.01, 1.10, 0.10, 0.01, key="taxa_montanha")
    with d2:
        impacto("o tamanho do passo. Abaixo de 1.0 o peso se aproxima do ideal. Em **1.0** ele fica pulando de um lado para o outro "
                "sem nunca chegar. **Acima de 1.0** ele se afasta mais a cada passo e o erro explode.")
    ep_m = d3.slider("Épocas (passos)", 1, 40, 10, key="ep_montanha")
    with d3:
        impacto("quantos passos damos. Com taxa pequena você precisa de **muitas** épocas. Com taxa boa, poucas bastam. "
                "As duas escolhas trabalham juntas.")

    caminho = descer_montanha(w0, taxa_m, ep_m)
    fator = abs(1 - 2 * taxa_m)
    perdas = (caminho - 3.0) ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    grade = np.linspace(-12, 18, 300)
    ax1.plot(grade, (grade - 3) ** 2, color="gray")
    dentro = np.abs(caminho - 3) < 15
    ax1.plot(caminho[dentro], perdas[dentro], "o-", color="tab:red", markersize=4)
    ax1.plot(caminho[0], perdas[0], "s", color="tab:green", markersize=9, label="início")
    ax1.set_xlim(-12, 18)
    ax1.set_ylim(0, 150)
    ax1.set_xlabel("valor do peso w"); ax1.set_ylabel("loss")
    ax1.set_title("A montanha e o caminho do peso", fontsize=10)
    ax1.legend()
    ax2.plot(range(len(perdas)), perdas + 1e-12, "o-", color="tab:blue", markersize=4)
    ax2.set_yscale("log")
    ax2.set_xlabel("época"); ax2.set_ylabel("loss (escala log)")
    ax2.set_title("O erro a cada época", fontsize=10)
    st.pyplot(fig); plt.close(fig)

    if fator > 1:
        st.error(f"Taxa grande demais: a cada época a distância até o ideal é multiplicada por {fator:.2f} (maior que 1). "
                 "O peso se afasta cada vez mais. Isso é a **divergência**.")
    elif abs(fator - 1) < 1e-9:
        st.warning("Taxa exatamente no limite: o peso fica pulando de um lado ao outro do vale, sem nunca descer.")
    else:
        st.success(f"A cada época, a distância até o ideal é multiplicada por **{fator:.2f}**. "
                   f"Depois de {ep_m} épocas, w = {caminho[-1]:.4f} (o ideal é 3).")
    st.write("Veja a conta, época por época:")
    st.dataframe(pd.DataFrame({"w": caminho.round(4),
                               "gradiente = 2 x (w - 3)": (2 * (caminho - 3)).round(4),
                               "loss": perdas.round(4)}).rename_axis("época (0 = início)").head(15))
    crianca("Descer o vale com passos minúsculos é seguro, mas leva a tarde inteira. Com passos gigantes você atravessa o vale "
            "e cai na encosta do outro lado, cada vez mais alto. O passo certo desce rápido sem passar do ponto.")
    mostrar_codigo(descer_montanha, titulo="Ver o código-fonte da descida da montanha")

    # ---------- 8.4 conta total ----------
    st.subheader("8.4 Quanta matemática cabe em um treino?")
    st.write("Lembra do treino da aba 7? Vamos contar as operações. Em cada amostra, a rede faz 9 multiplicações para o palpite "
             "e mais 9 para o gradiente.")
    ops_epoca = n_tr * 9 * 2
    ops_total = ops_epoca * max(epocas_feitas, 1)
    o1, o2, o3 = st.columns(3)
    o1.metric("Multiplicações por época", f"{ops_epoca:,}".replace(",", "."))
    o2.metric("Épocas feitas", max(epocas_feitas, 1))
    o3.metric("Multiplicações no treino todo", f"{ops_total:,}".replace(",", "."))
    st.caption(f"Conta: {n_tr} amostras x 18 multiplicações = {ops_epoca:,} por época; vezes {max(epocas_feitas, 1)} épocas."
               .replace(",", "."))
    st.write("Parece muito para só 9 pesos? Uma rede famosa como a ResNet-50 tem na ordem de **25 milhões de pesos** e faz na ordem de "
             "**4 bilhões de multiplicações por imagem**, em cada época, para **milhões de imagens**. O mecanismo é o mesmo "
             "que você acabou de ver, só que em escala gigante.")
    crianca("Se um pedreiro assenta 9 tijolos para fazer um muro pequeno, o arranha-céu é só o mesmo gesto repetido bilhões de vezes. "
            "Não tem um tijolo mágico.")

    # ---------- 8.5 resumo ----------
    st.subheader("8.5 A história completa em uma tabela")
    st.markdown("""
| Peça | O que ela faz na matemática | Se estiver mal ajustada |
|---|---|---|
| **Pesos e bias** | São os números que a rede aprende | Palpites ruins |
| **Perceptron / convolução** | Soma ponderada de entradas | Não detecta o padrão certo |
| **ReLU** | Zera os negativos, quebra a linearidade | Neurônios 'dormem' ou a rede vira uma conta só |
| **Loss** | Mede o tamanho do erro com uma fórmula | A rede não sabe o que melhorar |
| **Gradiente** | Mostra para que lado cada peso deve mexer | O treino anda sem direção |
| **Taxa de aprendizado** | Tamanho do passo de cada ajuste | Lenta demais ou explode |
| **Batch** | Quantas amostras entram em cada ajuste | Ajustes ruidosos ou poucos ajustes |
| **Época** | Uma volta completa pelos dados de treino | Poucas: aprende pouco. Demais: decora (overfitting) |
    """)
    st.success("**Conclusão:** uma rede neural não 'entende' a foto. Ela começa com números aleatórios, mede o erro com uma fórmula, "
               "calcula para que lado cada número deve andar e dá um passinho. Repete isso milhares de vezes. "
               "Quando a gente diz que a rede 'aprendeu', o que aconteceu foi **o erro diminuiu**. "
               "Sem mágica, com muita matemática.")
    experimente("Na descida da montanha, use taxa 0.05 e 40 épocas, depois taxa 0.4 e só 5 épocas. As duas chegam perto do ideal.",
                "Suba a taxa para 1.0 e depois 1.05, e compare com o que aconteceu na aba 7 com taxa 1.0.",
                "Volte à aba 7 e tente prever, antes de mexer, se o treino vai convergir. Depois confira.")
