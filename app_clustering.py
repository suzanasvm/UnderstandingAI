# -*- coding: utf-8 -*-
"""
Laboratorio Virtual de IA: Agrupamento (Clustering)
Aplicacao educacional que mostra, passo a passo e de forma visual,
como funciona o Aprendizado NAO Supervisionado, usando o algoritmo
K-Means da biblioteca scikit-learn.

Publico-alvo: estudantes iniciantes (ensino medio, tecnico ou superior).

Caracteristicas:
- Dados padrao embutidos: 16 pessoas que usam Netflix, com 3 variaveis
  numericas (Idade, Horas por Semana e Series Iniciadas por Mes).
- Se existir um arquivo usuarios_netflix.csv na mesma pasta, ele e usado
  no lugar dos dados embutidos.
- O usuario tambem pode enviar qualquer outro CSV pela barra lateral,
  e o laboratorio se adapta sozinho as novas colunas.
- Nao ha rotulos (labels): o algoritmo descobre os grupos sozinho.
- Animacao mostra os centroides se movendo a cada iteracao do K-Means.

Como executar:
    streamlit run app_clustering.py
"""

import io
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# Constantes gerais
# ----------------------------------------------------------------------------
ARQUIVO_CSV_PADRAO = Path(__file__).parent / "usuarios_netflix.csv"
MAX_ROTULOS_GRAFICO = 20   # maximo de linhas para escrever nomes no grafico
PALETA = ["#e4572e", "#2e86ab", "#59a96a", "#9b5de5", "#f4a259", "#3aa4ad", "#b05687", "#6c757d"]

# Dados padrao embutidos no codigo: pessoas que usam Netflix.
# Repare que NAO existe nenhuma coluna de resposta (label).
# E exatamente esse o ponto do aprendizado nao supervisionado!
DADOS_EMBUTIDOS_CSV = """Usuário,Idade,Horas por Semana,Séries Iniciadas por Mês
Alice,17,32,11
Bianca,19,28,9
Caio,22,35,12
Davi,24,26,8
Elisa,16,30,10
Fábio,45,4,1
Gustavo,50,6,2
Heloísa,38,3,1
Ivan,42,8,3
Joana,47,5,2
Karen,28,15,5
Lucas,31,18,6
Marina,26,12,4
Nando,33,16,5
Olívia,29,14,4
Paulo,35,13,4
"""

ETAPAS = [
    "1. Introdução",
    "2. Os dados",
    "3. Escolhendo o K",
    "4. O algoritmo em ação",
    "5. Grupos e centróides",
    "6. Experimente!",
]

# ----------------------------------------------------------------------------
# Configuracao da pagina e estilo visual
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laboratório de Clustering",
    page_icon="🍿",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #3a1d1d; }

    .stApp { background: linear-gradient(180deg, #fdf7f2 0%, #f9efe6 100%); }

    /* Cartoes didaticos */
    .cartao {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 20px;
        border: 1px solid #f0dfd2;
        box-shadow: 0 4px 14px rgba(90, 45, 20, .06);
        height: 100%;
        transition: transform .25s ease, box-shadow .25s ease;
        animation: surgir .6s ease both;
    }
    .cartao:hover { transform: translateY(-4px); box-shadow: 0 10px 22px rgba(90,45,20,.12); }
    .cartao .emoji { font-size: 34px; }
    .cartao h4 { margin: 8px 0 6px 0; color: #a63d2a; font-family: 'Fraunces', serif; }
    .cartao p { font-size: 14px; color: #6b4b3e; line-height: 1.55; margin: 0; }

    @keyframes surgir {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Caixa de destaque explicativa */
    .explica {
        background: #fdf1e7;
        border-left: 5px solid #e4572e;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        color: #5c3a2e;
        font-size: 15px;
        line-height: 1.6;
        margin: 10px 0;
    }

    /* Etiquetas coloridas */
    .chip {
        display: inline-block;
        padding: 7px 14px;
        margin: 4px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
        animation: surgir .5s ease both;
        color: #ffffff;
    }

    /* Barra de etapas */
    .stepper { display:flex; gap:6px; flex-wrap:wrap; margin-bottom: 4px; }
    .step {
        padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 600;
        background: #f0e2d6; color: #a08574;
    }
    .step.ativa { background: #a63d2a; color: #ffffff; }
    .step.feita { background: #f2c1a0; color: #7c3a1d; }

    div.stButton > button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# FUNCOES DE DADOS
# ============================================================================
def carregar_dados_padrao() -> pd.DataFrame:
    """
    Carrega os dados padrao. Prioridade:
    1. Arquivo usuarios_netflix.csv na mesma pasta (facil de editar)
    2. Dados embutidos no proprio codigo (garante que o app sempre funciona)
    """
    if ARQUIVO_CSV_PADRAO.exists():
        return pd.read_csv(ARQUIVO_CSV_PADRAO)
    return pd.read_csv(io.StringIO(DADOS_EMBUTIDOS_CSV))


def detectar_colunas(df: pd.DataFrame) -> dict:
    """
    Observa o CSV e descobre o papel de cada coluna:
      - id: coluna de texto em que cada linha tem um valor diferente
            (um nome ou codigo que apenas identifica o registro)
      - features: todas as colunas numericas (o que o algoritmo compara)
    Repare que aqui NAO procuramos um label: no aprendizado nao
    supervisionado, nao existe coluna de resposta!
    """
    numericas = df.select_dtypes(include="number").columns.tolist()
    texto = [c for c in df.columns if c not in numericas]

    col_id = None
    for c in texto:
        if df[c].nunique(dropna=True) == len(df):
            col_id = c
            break

    return {"id": col_id, "features": numericas}


def obter_dados():
    """Garante que o dataframe e a deteccao de colunas estejam na sessao."""
    if "df" not in st.session_state:
        st.session_state.df = carregar_dados_padrao()
        st.session_state.origem = "usuarios_netflix.csv (padrão)"
    if "colunas" not in st.session_state:
        st.session_state.colunas = detectar_colunas(st.session_state.df)
    if "features" not in st.session_state:
        # Por padrao, usamos as duas primeiras variaveis numericas,
        # porque duas variaveis cabem perfeitamente num grafico.
        st.session_state.features = st.session_state.colunas["features"][:2]
    return st.session_state.df, st.session_state.colunas


def trocar_dados(novo_df: pd.DataFrame, origem: str):
    """Substitui os dados, refaz a deteccao e zera o progresso do algoritmo."""
    st.session_state.df = novo_df
    st.session_state.origem = origem
    st.session_state.colunas = detectar_colunas(novo_df)
    st.session_state.features = st.session_state.colunas["features"][:2]
    st.session_state.modelo = None
    st.session_state.executado = False
    for chave in list(st.session_state.keys()):
        if str(chave).startswith(("sl_", "sel_")):
            del st.session_state[chave]


def nomes_das_linhas(df, col_id):
    """Usa a coluna de identificacao se existir; senao, numera os registros."""
    if col_id:
        return df[col_id].astype(str)
    return pd.Series([f"Registro {i + 1}" for i in range(len(df))], index=df.index)


# ============================================================================
# FUNCOES DO ALGORITMO
# ============================================================================
def kmeans_passo_a_passo(X: np.ndarray, k: int, semente: int, max_iteracoes: int = 10):
    """
    Reproduz o K-Means de forma didatica, guardando cada iteracao
    para podermos ANIMAR o aprendizado na tela.

    O ciclo do K-Means e simples e se repete ate estabilizar:
      1. Sortear k centroides iniciais (um chute)
      2. Cada ponto entra no grupo do centroide mais PROXIMO
      3. Cada centroide se move para o CENTRO do seu grupo
      4. Voltar ao passo 2, ate ninguem mais mudar de lugar
    """
    rng = np.random.default_rng(semente)
    indices_iniciais = rng.choice(len(X), size=k, replace=False)
    centroides = X[indices_iniciais].astype(float)

    historico = []
    for _ in range(max_iteracoes):
        # Passo de ATRIBUICAO: distancia de cada ponto a cada centroide
        distancias = np.linalg.norm(X[:, None, :] - centroides[None, :, :], axis=2)
        grupos = distancias.argmin(axis=1)
        historico.append({"centroides": centroides.copy(), "grupos": grupos.copy()})

        # Passo de ATUALIZACAO: centroide vira a media do seu grupo
        novos = np.array([
            X[grupos == j].mean(axis=0) if np.any(grupos == j) else centroides[j]
            for j in range(k)
        ])
        if np.allclose(novos, centroides):
            break  # nada mudou: o algoritmo convergiu!
        centroides = novos

    return historico


def treinar_kmeans_sklearn(X: np.ndarray, k: int, semente: int) -> KMeans:
    """
    Treina o modelo oficial com a biblioteca scikit-learn.
    E este modelo que usamos para classificar pontos novos na etapa final.
    """
    modelo = KMeans(n_clusters=k, random_state=semente, n_init=10)
    modelo.fit(X)
    return modelo


# ============================================================================
# FUNCOES DE INTERFACE
# ============================================================================
def ir_para(indice: int):
    st.session_state.etapa = indice


def barra_etapas():
    html = "<div class='stepper'>"
    for i, nome in enumerate(ETAPAS):
        classe = "ativa" if i == st.session_state.etapa else ("feita" if i < st.session_state.etapa else "")
        html += f"<div class='step {classe}'>{nome}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def botoes_navegacao(pode_avancar=True, aviso=""):
    st.divider()
    c1, _, c3 = st.columns([1, 4, 1])
    with c1:
        if st.session_state.etapa > 0:
            st.button("⬅️ Voltar", on_click=ir_para, args=(st.session_state.etapa - 1,), width='stretch')
    with c3:
        if st.session_state.etapa < len(ETAPAS) - 1:
            st.button("Avançar ➡️", on_click=ir_para, args=(st.session_state.etapa + 1,),
                      disabled=not pode_avancar, width='stretch', type="primary")
    if not pode_avancar and aviso:
        st.warning(aviso)


def cartao(emoji, titulo, texto):
    st.markdown(
        f"""
        <div class='cartao'>
            <div class='emoji'>{emoji}</div>
            <h4>{titulo}</h4>
            <p>{texto}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def explica(texto):
    st.markdown(f"<div class='explica'>{texto}</div>", unsafe_allow_html=True)


def grafico_dispersao(df, eixo_x, eixo_y, nomes, grupos=None, centroides=None,
                      ponto_novo=None, titulo=""):
    """
    Desenha o grafico de dispersao usado em varias etapas.
      - grupos: array com o grupo de cada ponto (colore os pontos)
      - centroides: matriz k x 2 com a posicao dos centroides (desenha X)
      - ponto_novo: tupla (x, y) com um ponto criado pelo estudante
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    fig.patch.set_facecolor("#ffffff")

    if grupos is None:
        # Antes do algoritmo rodar, todos os pontos tem a mesma cor:
        # nao sabemos (ainda!) quem pertence a qual grupo.
        ax.scatter(df[eixo_x], df[eixo_y], s=130, color="#8d99ae",
                   alpha=.85, edgecolors="white", linewidths=1.5)
    else:
        for j in sorted(set(grupos)):
            mascara = grupos == j
            ax.scatter(df[eixo_x][mascara], df[eixo_y][mascara], s=130,
                       color=PALETA[j % len(PALETA)], alpha=.85,
                       edgecolors="white", linewidths=1.5, label=f"Grupo {j + 1}")

    if centroides is not None:
        ax.scatter(centroides[:, 0], centroides[:, 1], s=420, marker="X",
                   color=[PALETA[j % len(PALETA)] for j in range(len(centroides))],
                   edgecolors="#222222", linewidths=2, zorder=5, label="Centróides")

    if ponto_novo is not None:
        ax.scatter([ponto_novo[0]], [ponto_novo[1]], s=300, marker="*",
                   color="#ffd60a", edgecolors="#222222", linewidths=1.5,
                   zorder=6, label="Seu ponto")

    if len(df) <= MAX_ROTULOS_GRAFICO:
        for idx in df.index:
            ax.annotate(str(nomes.loc[idx]), (df[eixo_x].loc[idx], df[eixo_y].loc[idx]),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8, color="#555")

    ax.set_xlabel(eixo_x)
    ax.set_ylabel(eixo_y)
    if titulo:
        ax.set_title(titulo, fontsize=12, color="#3a1d1d")
    if grupos is not None or centroides is not None or ponto_novo is not None:
        ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=.25)
    return fig


def limites_do_controle(df, coluna):
    """Calcula limites amigaveis para os sliders a partir dos dados reais."""
    serie = df[coluna].dropna()
    eh_inteira = bool((serie % 1 == 0).all())
    if eh_inteira:
        minimo = min(0, int(math.floor(serie.min())))
        maximo = int(math.ceil(serie.max() * 1.25)) or 1
        padrao = int(round(serie.median()))
        return minimo, maximo, padrao, 1
    minimo = float(min(0.0, serie.min()))
    maximo = float(serie.max() * 1.25)
    padrao = float(serie.median())
    passo = round((maximo - minimo) / 100, 4) or 0.01
    return minimo, maximo, padrao, passo


# ============================================================================
# ESTADO DA APLICACAO
# ============================================================================
if "etapa" not in st.session_state:
    st.session_state.etapa = 0
if "k" not in st.session_state:
    st.session_state.k = 3
if "semente" not in st.session_state:
    st.session_state.semente = 7
if "modelo" not in st.session_state:
    st.session_state.modelo = None
if "executado" not in st.session_state:
    st.session_state.executado = False

DADOS, COLUNAS = obter_dados()
COL_ID = COLUNAS["id"]
FEATURES_DISPONIVEIS = COLUNAS["features"]
NOMES = nomes_das_linhas(DADOS, COL_ID)


# ============================================================================
# BARRA LATERAL: FONTE DOS DADOS
# ============================================================================
with st.sidebar:
    st.header("📂 Fonte dos dados")
    st.markdown(
        f"Dados em uso: **{st.session_state.origem}**\n\n"
        f"Registros: **{len(DADOS)}** | Variáveis numéricas: **{len(FEATURES_DISPONIVEIS)}**"
    )
    st.caption(
        "Por padrão, o laboratório usa dados de pessoas que assistem Netflix. "
        "Mas você pode enviar qualquer CSV: o app se adapta sozinho às novas colunas."
    )
    enviado = st.file_uploader("Enviar meu CSV", type=["csv"])
    if enviado is not None and st.session_state.origem != enviado.name:
        try:
            novo = pd.read_csv(enviado)
            trocar_dados(novo, enviado.name)
            st.success(f"Dados de {enviado.name} carregados!")
            st.rerun()
        except Exception as erro:
            st.error(f"Não consegui ler esse CSV: {erro}")
    if not st.session_state.origem.endswith("(padrão)"):
        if st.button("↩️ Voltar aos dados padrão"):
            trocar_dados(carregar_dados_padrao(), "usuarios_netflix.csv (padrão)")
            st.rerun()


# ============================================================================
# CABECALHO
# ============================================================================
st.title("🍿Modelo Não Supervisionado - Laboratório de Clustering")
st.caption(
    "Veja, passo a passo, como um algoritmo de Aprendizado Não Supervisionado "
    "descobre grupos escondidos nos dados, sem que ninguém diga a resposta."
)
barra_etapas()
st.write("")

etapa = st.session_state.etapa

# Validacao minima: precisamos de pelo menos 2 variaveis numericas
if etapa > 0 and len(FEATURES_DISPONIVEIS) < 2:
    st.error(
        "O CSV atual precisa de pelo menos duas colunas numéricas para o "
        "laboratório funcionar. Envie outro arquivo pela barra lateral."
    )
    st.stop()


# ============================================================================
# ETAPA 1 - INTRODUCAO
# ============================================================================
if etapa == 0:
    st.header("Bem-vindo ao laboratório! 👋")
    explica(
        "Nos laboratórios de aprendizado <b>supervisionado</b>, o modelo aprende com exemplos "
        "que já vêm com a resposta certa, como um gabarito. Aqui será diferente: vamos entregar os dados "
        "<b>sem nenhuma resposta</b> e o algoritmo terá que descobrir sozinho quais grupos existem. "
        "Isso se chama <b>Aprendizado Não Supervisionado</b>."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        cartao("🧺", "Um exemplo do cotidiano",
               "Imagine despejar uma cesta de frutas misturadas na mesa e pedir para alguém organizá-las, "
               "<b>sem dizer os nomes das frutas</b>. A pessoa vai juntar as parecidas: as redondas e vermelhas "
               "num canto, as compridas e amarelas em outro. Ela criou grupos sem conhecer os rótulos!")
    with c2:
        cartao("🔍", "Agrupamento (Clustering)",
               "É exatamente essa tarefa: olhar para os dados e juntar os registros <b>parecidos entre si</b> "
               "em grupos, chamados de <b>clusters</b>. Empresas usam isso para agrupar clientes com gostos "
               "parecidos, escolas para identificar perfis de estudantes, e serviços de streaming para "
               "entender tipos de espectadores.")
    with c3:
        cartao("📏", "Similaridade é distância",
               "Como o computador sabe se dois registros são parecidos? Colocando cada um como um ponto num "
               "gráfico e medindo a <b>distância</b> entre eles. Pontos próximos são parecidos; pontos "
               "distantes são diferentes. Simples assim!")

    st.write("")
    c4, c5 = st.columns(2)
    with c4:
        cartao("🎯", "O algoritmo K-Means",
               "É o algoritmo de agrupamento mais famoso do mundo. O <b>K</b> é o número de grupos que "
               "pedimos para ele encontrar, e <b>Means</b> significa médias, porque cada grupo é representado "
               "pelo seu ponto central, o <b>centróide</b>. Você vai ver os centróides se movendo ao vivo na etapa 4!")
    with c5:
        cartao("🍿", "Nossa missão",
               "Temos dados de pessoas que assistem Netflix: idade, horas por semana e séries iniciadas por mês. "
               "<b>Ninguém nos disse quais perfis de espectador existem.</b> Será que o K-Means consegue "
               "descobrir sozinho? Vamos acompanhar cada passo!")

    botoes_navegacao()

# ============================================================================
# ETAPA 2 - APRESENTACAO E EXPLORACAO DOS DADOS
# ============================================================================
elif etapa == 1:
    st.header("📊 Etapa 2: Conhecendo os dados")

    explica(
        f"O laboratório carregou <b>{len(DADOS)} registros</b>. Repare numa coisa importante: "
        "<b>não existe nenhuma coluna de resposta</b>, nenhum rótulo dizendo o grupo de cada pessoa. "
        "É essa a grande diferença para o aprendizado supervisionado. O algoritmo vai receber apenas "
        "os números e terá que encontrar os padrões por conta própria."
    )

    d1, d2 = st.columns(2)
    with d1:
        cartao("🪪", "Identificação",
               (f"Coluna <b>{COL_ID}</b>: cada linha tem um valor diferente, então ela só serve para "
                "identificar o registro. O algoritmo não usa essa coluna.")
               if COL_ID else
               "Nenhuma coluna de identificação foi encontrada. Os registros serão numerados automaticamente.")
    with d2:
        nomes_feats = ", ".join(f"<b>{f}</b>" for f in FEATURES_DISPONIVEIS)
        cartao("🔢", "Variáveis numéricas",
               f"Colunas encontradas: {nomes_feats}. É com elas que o algoritmo vai medir a "
               "similaridade entre os registros. E não há coluna de rótulo: aqui ninguém dá a resposta!")

    st.write("")
    st.dataframe(DADOS, width='stretch', hide_index=True, height=300)

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("Escolha as variáveis do gráfico")
        explica(
            "Para enxergar os dados, vamos usar <b>duas variáveis</b> por vez, uma em cada eixo. "
            "Escolha abaixo quais comparar. Dica: experimente combinações diferentes e observe "
            "quando os pontos formam grupinhos visíveis!"
        )
        eixo_x = st.selectbox("Eixo horizontal", FEATURES_DISPONIVEIS, index=0, key="sel_x")
        opcoes_y = [f for f in FEATURES_DISPONIVEIS if f != eixo_x]
        eixo_y = st.selectbox("Eixo vertical", opcoes_y, index=0, key="sel_y")
        # Guardamos a escolha: o algoritmo vai agrupar usando essas duas variaveis
        if [eixo_x, eixo_y] != st.session_state.features:
            st.session_state.features = [eixo_x, eixo_y]
            st.session_state.executado = False
    with c2:
        fig = grafico_dispersao(DADOS, eixo_x, eixo_y, NOMES,
                                titulo="Cada ponto é uma pessoa. Todas da mesma cor: ainda não há grupos!")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    explica(
        "👀 <b>Olhe com atenção para o gráfico.</b> Consegue perceber pontos que ficam juntinhos, formando "
        "regiões mais cheias? Seu cérebro acabou de fazer clustering visualmente! Na próxima etapa, "
        "vamos decidir quantos grupos pedir para o algoritmo encontrar."
    )
    botoes_navegacao()

# ============================================================================
# ETAPA 3 - ESCOLHA DO NUMERO DE GRUPOS (K)
# ============================================================================
elif etapa == 2:
    st.header("🎚️ Etapa 3: Quantos grupos existem? Escolhendo o K")

    explica(
        "O K-Means tem uma característica curiosa: <b>ele não descobre quantos grupos existem</b>. "
        "Quem decide isso somos nós, escolhendo o valor de <b>K</b>. Se K = 3, ele encontrará exatamente "
        "3 grupos; se K = 5, encontrará 5, mesmo que faça menos sentido. Parte do trabalho de quem estuda "
        "os dados é testar valores diferentes e comparar os resultados."
    )

    c1, c2 = st.columns(2)
    with c1:
        cartao("🛒", "Pensando com um exemplo",
               "Um mercado quer agrupar clientes. Com K = 2, talvez encontre 'quem gasta muito' e 'quem gasta pouco'. "
               "Com K = 3, pode surgir um grupo intermediário. Com K = 10, os grupos ficam tão pequenos que "
               "param de ajudar. Não existe resposta única: existe a escolha que melhor explica os dados.")
    with c2:
        cartao("💡", "Uma boa pista",
               "Olhe o gráfico da etapa anterior: quantos aglomerados de pontos os seus olhos percebem? "
               "Essa é uma ótima primeira aposta para o K. Depois, você pode voltar aqui, mudar o valor e "
               "rodar o algoritmo de novo para comparar!")

    st.write("")
    k_maximo = min(8, len(DADOS) - 1)
    k = st.slider("Escolha o valor de K (número de grupos)", 2, k_maximo, min(st.session_state.k, k_maximo),
                  help="O algoritmo vai procurar exatamente essa quantidade de grupos.")
    if k != st.session_state.k:
        st.session_state.k = k
        st.session_state.executado = False

    marcadores = "".join(
        f"<span class='chip' style='background:{PALETA[j % len(PALETA)]}'>Grupo {j + 1}</span>"
        for j in range(k)
    )
    st.markdown(f"O algoritmo vai procurar estes {k} grupos: {marcadores}", unsafe_allow_html=True)

    explica(
        f"✅ K definido como <b>{k}</b>. Na próxima etapa, você verá o algoritmo trabalhando ao vivo: "
        "os centróides nascem em posições sorteadas e vão se movendo, rodada após rodada, "
        "até cada um encontrar o centro do seu grupo."
    )
    botoes_navegacao()

# ============================================================================
# ETAPA 4 - EXECUCAO DO ALGORITMO COM ANIMACAO
# ============================================================================
elif etapa == 3:
    st.header("⚙️ Etapa 4: O K-Means em ação")

    eixo_x, eixo_y = st.session_state.features
    X = DADOS[[eixo_x, eixo_y]].to_numpy(dtype=float)

    explica(
        "O K-Means funciona repetindo um ciclo simples, como uma dança em dois passos:<br><br>"
        "1️⃣ <b>Sorteio:</b> os K centróides nascem em posições aleatórias (é só um chute inicial).<br>"
        "2️⃣ <b>Atribuição:</b> cada ponto entra no grupo do centróide mais <b>próximo</b> dele.<br>"
        "3️⃣ <b>Atualização:</b> cada centróide se move para o <b>centro (média)</b> do seu grupo.<br>"
        "4️⃣ <b>Repetição:</b> os passos 2 e 3 se repetem até ninguém mais mudar de grupo.<br><br>"
        "Aperte o botão e observe os X gigantes (os centróides) se movendo a cada rodada!"
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        rodar = st.button(f"🚀 Rodar o K-Means com K = {st.session_state.k}", type="primary", width='stretch')
    with c2:
        if st.button("🎲 Sortear novas posições iniciais", width='stretch'):
            st.session_state.semente += 1
            st.session_state.executado = False
            st.info("Novas posições sorteadas! Rode o algoritmo de novo e compare o caminho dos centróides.")

    if rodar:
        area = st.empty()
        status = st.empty()

        # Animacao didatica: reproduzimos cada iteracao do K-Means
        historico = kmeans_passo_a_passo(X, st.session_state.k, st.session_state.semente)
        for i, quadro in enumerate(historico):
            if i == 0:
                status.info(f"🎲 Rodada {i + 1}: centróides sorteados. Cada ponto entra no grupo do centróide mais próximo.")
            elif i == len(historico) - 1:
                status.success(f"🏁 Rodada {i + 1}: ninguém mais mudou de grupo. O algoritmo convergiu!")
            else:
                status.info(f"🔄 Rodada {i + 1}: os centróides se moveram para o centro dos seus grupos e os pontos foram reatribuídos.")
            with area.container():
                fig = grafico_dispersao(
                    DADOS, eixo_x, eixo_y, NOMES,
                    grupos=quadro["grupos"], centroides=quadro["centroides"],
                    titulo=f"Rodada {i + 1} de {len(historico)}",
                )
                st.pyplot(fig, width='stretch')
                plt.close(fig)
            time.sleep(1.3)

        # Modelo oficial da scikit-learn, usado nas proximas etapas
        st.session_state.modelo = treinar_kmeans_sklearn(X, st.session_state.k, st.session_state.semente)
        st.session_state.executado = True
        st.balloons()

    if st.session_state.executado and st.session_state.modelo is not None:
        explica(
            f"🎉 O algoritmo encontrou os <b>{st.session_state.k} grupos</b> e parou sozinho quando nada mais mudava "
            "(isso se chama <b>convergência</b>). Quer investigar? Volte à etapa 3, mude o K, ou clique em "
            "'Sortear novas posições iniciais' e rode de novo para comparar os resultados. Quando estiver "
            "satisfeito, avance para explorar os grupos em detalhe!"
        )

    botoes_navegacao(pode_avancar=st.session_state.executado,
                     aviso="Rode o algoritmo pelo menos uma vez antes de avançar.")

# ============================================================================
# ETAPA 5 - GRUPOS, CENTROIDES E ATRIBUICOES
# ============================================================================
elif etapa == 4:
    st.header("🗺️ Etapa 5: Explorando os grupos e os centróides")

    if not st.session_state.executado or st.session_state.modelo is None:
        st.warning("O algoritmo ainda não foi executado. Volte para a etapa 4.")
        botoes_navegacao(pode_avancar=False, aviso="")
    else:
        eixo_x, eixo_y = st.session_state.features
        X = DADOS[[eixo_x, eixo_y]].to_numpy(dtype=float)
        modelo = st.session_state.modelo
        grupos = modelo.labels_
        centroides = modelo.cluster_centers_

        explica(
            "O resultado final está pronto! Cada cor é um grupo descoberto pelo algoritmo, e cada <b>X</b> "
            "é um <b>centróide</b>: o ponto médio do grupo, uma espécie de 'membro típico' que resume "
            "todo mundo ao redor dele."
        )

        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = grafico_dispersao(DADOS, eixo_x, eixo_y, NOMES,
                                    grupos=grupos, centroides=centroides,
                                    titulo="Resultado final do agrupamento")
            st.pyplot(fig, width='stretch')
            plt.close(fig)
        with c2:
            st.subheader("Posição dos centróides")
            tabela_c = pd.DataFrame(centroides, columns=[eixo_x, eixo_y]).round(1)
            tabela_c.insert(0, "Grupo", [f"Grupo {j + 1}" for j in range(len(centroides))])
            st.dataframe(tabela_c, width='stretch', hide_index=True)
            explica(
                "Leia assim: o centróide do Grupo 1 representa o membro 'médio' daquele grupo. "
                "Comparar os centróides é a forma mais rápida de entender o que diferencia cada perfil."
            )

        st.subheader("Quem ficou em cada grupo?")
        atribuicao = DADOS.copy()
        atribuicao["Grupo"] = [f"Grupo {g + 1}" for g in grupos]
        st.dataframe(atribuicao, width='stretch', hide_index=True, height=300)

        st.subheader("Retrato falado de cada grupo")
        medias = atribuicao.groupby("Grupo")[FEATURES_DISPONIVEIS].mean().round(1)
        medias["Quantidade"] = atribuicao.groupby("Grupo").size()
        st.dataframe(medias, width='stretch')
        explica(
            "💬 <b>Agora vem a parte humana do trabalho:</b> o algoritmo entrega os grupos, mas quem dá "
            "significado a eles somos nós. Olhe as médias acima e tente batizar cada grupo. Nos dados "
            "padrão da Netflix, costuma aparecer algo como 'jovens maratonistas', 'espectadores casuais' "
            "e 'público moderado'. Que nomes você daria?"
        )
        botoes_navegacao()

# ============================================================================
# ETAPA 6 - AREA DE EXPERIMENTACAO
# ============================================================================
elif etapa == 5:
    st.header("🧪 Etapa 6: Experimente! Crie um ponto novo")

    if not st.session_state.executado or st.session_state.modelo is None:
        st.warning("O algoritmo ainda não foi executado. Volte para a etapa 4.")
        botoes_navegacao(pode_avancar=False, aviso="")
    else:
        eixo_x, eixo_y = st.session_state.features
        modelo = st.session_state.modelo
        grupos = modelo.labels_
        centroides = modelo.cluster_centers_

        explica(
            "Os grupos estão formados. Agora imagine que uma <b>pessoa nova</b> apareceu: em qual grupo ela "
            "se encaixa? Mova os controles abaixo e veja a estrela ⭐ se deslocar no gráfico. O algoritmo "
            "calcula a distância dela até cada centróide e escolhe o mais próximo. É assim que serviços "
            "reais decidem, por exemplo, que recomendações mostrar para um usuário recém-chegado!"
        )

        c1, c2 = st.columns([1, 1.4])
        with c1:
            valores = {}
            for f in (eixo_x, eixo_y):
                minimo, maximo, padrao, passo = limites_do_controle(DADOS, f)
                valores[f] = st.slider(f, minimo, maximo, padrao, step=passo, key=f"sl_{f}")

            ponto = np.array([[valores[eixo_x], valores[eixo_y]]], dtype=float)
            grupo_previsto = int(modelo.predict(ponto)[0])
            cor = PALETA[grupo_previsto % len(PALETA)]
            st.markdown(
                f"<h3>Este ponto pertence ao: "
                f"<span class='chip' style='background:{cor}; font-size:20px'>Grupo {grupo_previsto + 1}</span></h3>",
                unsafe_allow_html=True,
            )

            st.subheader("📏 As distâncias explicam a escolha")
            distancias = np.linalg.norm(centroides - ponto, axis=1)
            passos = []
            for j, d in enumerate(distancias):
                marcador = " ⬅️ <b>mais próximo!</b>" if j == grupo_previsto else ""
                passos.append(f"Distância até o centróide do Grupo {j + 1}: <b>{d:.1f}</b>{marcador}")
            explica("<br>".join(passos) +
                    "<br><br>O ponto entra no grupo do centróide mais próximo. Similaridade, aqui, é pura distância!")

        with c2:
            fig = grafico_dispersao(DADOS, eixo_x, eixo_y, NOMES,
                                    grupos=grupos, centroides=centroides,
                                    ponto_novo=(valores[eixo_x], valores[eixo_y]),
                                    titulo="A estrela é o seu ponto. Em qual grupo ela cai?")
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        st.divider()
        st.subheader("🎓 O que você aprendeu neste laboratório")
        r1, r2, r3 = st.columns(3)
        with r1:
            cartao("🧩", "Agrupamento e similaridade",
                   "Agrupar é juntar registros parecidos sem conhecer respostas prévias. "
                   "A similaridade é medida pela <b>distância</b> entre os pontos: quanto mais perto, mais parecidos.")
        with r2:
            cartao("🎯", "Centróides e o ciclo do K-Means",
                   "Cada grupo é resumido pelo seu <b>centróide</b>, o ponto médio. O K-Means repete o ciclo "
                   "'atribuir pontos ➜ mover centróides' até estabilizar. Você viu essa dança acontecer ao vivo!")
        with r3:
            cartao("🔮", "Padrões sem respostas",
                   "O algoritmo nunca soube o 'grupo certo' de ninguém, e mesmo assim encontrou estrutura nos dados. "
                   "É esse o poder do aprendizado não supervisionado: <b>revelar padrões escondidos</b> "
                   "que nem sabíamos que existiam.")

        st.write("")
        explica(
            "Quer continuar explorando? Volte às etapas anteriores para trocar as variáveis do gráfico, "
            "testar outros valores de K ou sortear novas posições iniciais. E, pela barra lateral, "
            "envie um CSV seu, sobre qualquer assunto, para agrupar os seus próprios dados!"
        )
        botoes_navegacao()
