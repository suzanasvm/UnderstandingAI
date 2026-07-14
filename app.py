# -*- coding: utf-8 -*-
"""
Laboratorio Virtual de IA

"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

ARQUIVO_CSV_PADRAO = Path(__file__).parent / "alunos.csv"

# ----------------------------------------------------------------------------
# Configuracao geral da pagina
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laboratório Virtual de IA - Aprendizado",
    page_icon="🧪",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Estilo visual (CSS customizado)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #1d3b4a !important; }

    p { color: #3f5b66 !important; }
    
    [data-testid="stSidebarContent"] :is(h1, h2, h3, p) { color: #ffffff !important; }

    [data-testid="stBaseButton-primary"] :is(h1, h2, h3, p) { color: #ffffff !important; }

    [data-testid="stBaseButton-secondary"] :is(h1, h2, h3, p) { color: #ffffff !important; }

    .stApp { background: linear-gradient(180deg, #f6fbfc 0%, #eef5f7 100%) !important; }

    .stCheckbox label p { color: #000000 !important; }

    .cartao {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 20px;
        border: 1px solid #dcebee;
        box-shadow: 0 4px 14px rgba(29, 59, 74, 0.06);
        height: 100%;
        transition: transform .25s ease, box-shadow .25s ease;
        animation: surgir .6s ease both;
    }
    .cartao:hover { transform: translateY(-4px); box-shadow: 0 10px 22px rgba(29,59,74,.12); }
    .cartao .emoji { font-size: 34px; }
    .cartao h4 { margin: 8px 0 6px 0; color: #14606e; font-family: 'Fraunces', serif; }
    .cartao p { font-size: 14px; color: #3f5b66; line-height: 1.55; margin: 0; }

    @keyframes surgir {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .aluno-chip {
        display: inline-block;
        padding: 7px 14px;
        margin: 4px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
        animation: surgir .5s ease both;
    }
    .treino   { background: #dff3ea; color: #147a52; border: 1.5px solid #9fdcc2; }
    .teste    { background: #fdeede; color: #b25b12; border: 1.5px solid #f4cfa4; }
    .aprovado { background: #dff3ea; color: #147a52; }
    .reprovado{ background: #fde3e3; color: #b02a2a; }
    .neutro   { background: #e3edf0; color: #2b5866; }

    .explica {
        background: #eef7f9;
        border-left: 5px solid #2b8fa3;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        color: #24505c;
        font-size: 15px;
        line-height: 1.6;
        margin: 10px 0;
    }

    .stepper { display:flex; gap:6px; flex-wrap:wrap; margin-bottom: 4px; }
    .step {
        padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 600;
        background: #e3edf0; color: #6b8894;
    }
    .step.ativa { background: #14606e; color: #ffffff; }
    .step.feita { background: #bfe3d4; color: #14624a; }

    div.stButton > button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

ETAPAS = [
    "1. Introdução",
    "2. Os dados",
    "3. Divisão treino e teste",
    "4. Treinamento do modelo",
    "5. Avaliação",
    "6. Faça uma previsão",
]

# ----------------------------------------------------------------------------
# Leitura do CSV e deteccao automatica das colunas
# ----------------------------------------------------------------------------
def carregar_csv_padrao() -> pd.DataFrame:
    """Le o arquivo alunos.csv que acompanha a aplicacao."""
    if not ARQUIVO_CSV_PADRAO.exists():
        st.error(
            f"Arquivo de dados não encontrado: {ARQUIVO_CSV_PADRAO.name}. "
            "Coloque o CSV na mesma pasta do app.py ou envie um arquivo pela barra lateral."
        )
        st.stop()
    return pd.read_csv(ARQUIVO_CSV_PADRAO)


def detectar_colunas(df: pd.DataFrame) -> dict:
    """
    Observa o CSV e descobre o papel de cada coluna:
      - id: coluna de texto em que cada linha tem um valor diferente (o nome do aluno)
      - label: coluna de texto com poucas categorias repetidas (a resposta a prever)
      - features: todas as colunas numericas (as pistas que o modelo pode usar)
    """
    numericas = df.select_dtypes(include="number").columns.tolist()
    texto = [c for c in df.columns if c not in numericas]

    col_id = None
    col_label = None
    for c in texto:
        valores_unicos = df[c].nunique(dropna=True)
        if valores_unicos == len(df) and col_id is None:
            col_id = c                      # identifica cada linha: e o nome
        elif 2 <= valores_unicos <= max(2, len(df) // 2) and col_label is None:
            col_label = c                   # poucas categorias: e o rotulo

    # Planos B, caso o CSV fuja do padrao
    if col_label is None and texto:
        col_label = texto[-1]
    if col_label is None and numericas:
        col_label = numericas[-1]
        numericas = numericas[:-1]

    return {"id": col_id, "label": col_label, "features": numericas}


def obter_dados():
    """Garante que o dataframe e a deteccao de colunas estejam no estado da sessao."""
    if "df" not in st.session_state:
        st.session_state.df = carregar_csv_padrao()
        st.session_state.origem = ARQUIVO_CSV_PADRAO.name
    if "colunas" not in st.session_state:
        st.session_state.colunas = detectar_colunas(st.session_state.df)
    if "features" not in st.session_state:
        st.session_state.features = st.session_state.colunas["features"].copy()
    return st.session_state.df, st.session_state.colunas


def trocar_dados(novo_df: pd.DataFrame, origem: str):
    """Substitui os dados e refaz toda a deteccao, zerando o progresso do modelo."""
    st.session_state.df = novo_df
    st.session_state.origem = origem
    st.session_state.colunas = detectar_colunas(novo_df)
    st.session_state.features = st.session_state.colunas["features"].copy()
    st.session_state.modelo = None
    st.session_state.treinado = False
    # limpa checkboxes antigos de features
    for chave in list(st.session_state.keys()):
        if str(chave).startswith("chk_"):
            del st.session_state[chave]


# ----------------------------------------------------------------------------
# Estado da aplicacao
# ----------------------------------------------------------------------------
if "etapa" not in st.session_state:
    st.session_state.etapa = 0
if "pct_treino" not in st.session_state:
    st.session_state.pct_treino = 80
if "semente" not in st.session_state:
    st.session_state.semente = 42
if "modelo" not in st.session_state:
    st.session_state.modelo = None
if "treinado" not in st.session_state:
    st.session_state.treinado = False

DADOS, COLUNAS = obter_dados()
COL_ID = COLUNAS["id"]
LABEL = COLUNAS["label"]
FEATURES_DISPONIVEIS = COLUNAS["features"]


# ----------------------------------------------------------------------------
# Funcoes de apoio
# ----------------------------------------------------------------------------
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
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if st.session_state.etapa > 0:
            st.button("⬅️ Voltar", on_click=ir_para, args=(st.session_state.etapa - 1,), width='stretch')
    with c3:
        if st.session_state.etapa < len(ETAPAS) - 1:
            st.button(
                "Avançar ➡️",
                on_click=ir_para,
                args=(st.session_state.etapa + 1,),
                disabled=not pode_avancar,
                width='stretch',
                type="primary",
            )
    if not pode_avancar and aviso:
        st.warning(aviso)


def nomes_das_linhas():
    """Usa a coluna de identificacao se existir; senao, numera as linhas."""
    if COL_ID:
        return DADOS[COL_ID]
    return pd.Series([f"Linha {i + 1}" for i in range(len(DADOS))], index=DADOS.index)


def dividir_dados():
    X = DADOS[st.session_state.features]
    y = DADOS[LABEL]
    tamanho_teste = 1 - st.session_state.pct_treino / 100
    try:
        # stratify tenta manter a proporcao das classes nos dois grupos
        return train_test_split(
            X, y, nomes_das_linhas(),
            test_size=tamanho_teste,
            random_state=st.session_state.semente,
            stratify=y,
        )
    except ValueError:
        return train_test_split(
            X, y, nomes_das_linhas(),
            test_size=tamanho_teste,
            random_state=st.session_state.semente,
        )


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


def desenhar_arvore(modelo, features, titulo="Árvore de Decisão"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#ffffff")
    plot_tree(
        modelo,
        feature_names=features,
        class_names=[str(c) for c in modelo.classes_],
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=10,
        ax=ax,
    )
    ax.set_title(titulo, fontsize=13, color="#1d3b4a")
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def limites_do_slider(coluna: str):
    """Cria limites amigaveis para o slider a partir dos valores reais do CSV."""
    serie = DADOS[coluna]
    minimo = min(0, int(math.floor(serie.min())))
    maximo = int(math.ceil(serie.max() * 1.25))
    padrao = int(round(serie.median()))
    return minimo, maximo, padrao


def classe_positiva():
    """Descobre qual classe representa o resultado 'bom' para colorir a previsao."""
    for c in DADOS[LABEL].unique():
        if str(c).strip().lower() in ("aprovado", "aprovada", "sim", "yes", "1", "true"):
            return c
    return None

# ----------------------------------------------------------------------------
# Barra lateral: fonte dos dados
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Fonte dos dados")

    st.markdown(
        f"""
**Arquivo em uso:** `{st.session_state.origem}`

**Linhas:** {len(DADOS)}
**Colunas:** {len(DADOS.columns)}
"""
    )

    st.info(
        "Você pode utilizar o conjunto de dados padrão do laboratório ou experimentar com outro arquivo CSV."
    )

    st.markdown(
        """
### 📥 Baixar o conjunto de dados

Caso deseje utilizar o mesmo conjunto de dados apresentado neste laboratório, faça o download do arquivo **alunos.csv** no GitHub:

🔗 https://github.com/suzanasvm/UnderstandingAI/blob/main/alunos.csv

Depois de baixar o arquivo, envie-o utilizando o campo abaixo.
"""
    )

    enviado = st.file_uploader(
        "Enviar um arquivo CSV",
        type=["csv"],
        help="Selecione um arquivo CSV para substituir a base de dados do laboratório."
    )

    if enviado is not None and st.session_state.origem != enviado.name:
        try:
            novo = pd.read_csv(enviado)
            trocar_dados(novo, enviado.name)
            st.success(f"Dados de {enviado.name} carregados com sucesso!")
            st.rerun()
        except Exception as erro:
            st.error(f"Não foi possível ler o arquivo CSV: {erro}")

    if st.session_state.origem != ARQUIVO_CSV_PADRAO.name:
        st.divider()

        if st.button("↩️ Voltar ao CSV original", width='stretch'):
            trocar_dados(carregar_csv_padrao(), ARQUIVO_CSV_PADRAO.name)
            st.rerun()
# ----------------------------------------------------------------------------
# Cabecalho
# ----------------------------------------------------------------------------
st.title("🧪 Laboratório Virtual de IA - Aprendizado Supervisionado")
st.caption(
    "Aprenda, passo a passo, como um modelo de Inteligência Artificial é treinado. "
    f"Os dados vêm do arquivo {st.session_state.origem}."
)
barra_etapas()
st.write("")

etapa = st.session_state.etapa

# Validacao minima do CSV para as etapas de dados em diante
if etapa > 0 and (LABEL is None or len(FEATURES_DISPONIVEIS) == 0):
    st.error(
        "O CSV atual não tem o formato esperado. Ele precisa de pelo menos "
        "uma coluna numérica (features) e uma coluna de categorias (label)."
    )
    st.stop()

# ============================================================================
# ETAPA 1 - INTRODUCAO
# ============================================================================
if etapa == 0:
    st.header("Bem-vindo ao laboratório! 👋")
    explica(
        "Aqui você vai <b>ver a IA aprendendo</b> na sua frente. Nada de fórmulas complicadas: "
        "vamos usar um exemplo simples, com notas de alunos lidas de um arquivo CSV, "
        "e acompanhar cada etapa do processo. Antes de começar, vamos entender 5 ideias fundamentais."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        cartao("🤖", "O que é IA?",
               "Inteligência Artificial é a área que cria programas capazes de realizar tarefas que normalmente exigem inteligência humana, como reconhecer padrões, tomar decisões e fazer previsões.")
    with c2:
        cartao("📚", "O que é Machine Learning?",
               "É um ramo da IA em que o computador <b>aprende com exemplos</b>, em vez de seguir regras escritas por uma pessoa. Quanto mais exemplos de qualidade, melhor ele aprende.")
    with c3:
        cartao("👩‍🏫", "Aprendizado Supervisionado",
               "É como estudar com gabarito: mostramos ao computador vários exemplos <b>junto com a resposta certa</b>. Ele analisa tudo e descobre sozinho os padrões que levam a cada resposta.")

    st.write("")
    c4, c5 = st.columns(2)
    with c4:
        cartao("🏷️", "Exemplos rotulados",
               "Cada linha do nosso CSV é um exemplo. O <b>rótulo</b> (label) é a resposta certa daquele exemplo. "
               f"No nosso caso, o rótulo é a coluna <b>{LABEL}</b>.")
    with c5:
        cartao("🏋️", "Treinar um modelo",
               "Treinar é o processo em que o algoritmo olha os exemplos rotulados e ajusta suas regras internas até conseguir acertar as respostas. O resultado é o <b>modelo</b>: uma espécie de cérebro treinado, pronto para prever casos novos.")

    st.write("")
    explica(
        f"🎯 <b>Nossa missão neste laboratório:</b> treinar um modelo que aprenda, a partir dos dados do arquivo "
        f"<b>{st.session_state.origem}</b>, a prever a coluna <b>{LABEL}</b> de alunos que ele nunca viu."
    )
    botoes_navegacao()

# ============================================================================
# ETAPA 2 - VISUALIZACAO DOS DADOS
# ============================================================================
elif etapa == 1:
    st.header("📊 Etapa 2: Conhecendo os dados")

    explica(
        f"O laboratório abriu o arquivo <b>{st.session_state.origem}</b>, encontrou "
        f"<b>{len(DADOS)} exemplos</b> e observou cada coluna para descobrir o papel dela. Veja o que foi detectado:"
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        cartao("🪪", "Identificação",
               (f"Coluna <b>{COL_ID}</b>: cada linha tem um valor diferente, então ela serve apenas para "
                "identificar o aluno. O modelo <b>não</b> usa essa coluna para aprender.")
               if COL_ID else
               "Nenhuma coluna de identificação foi encontrada. As linhas serão numeradas automaticamente.")
    with d2:
        nomes_feats = ", ".join(f"<b>{f}</b>" for f in FEATURES_DISPONIVEIS)
        cartao("🔢", "Entradas (features)",
               f"Colunas numéricas encontradas: {nomes_feats}. São as pistas que o modelo pode usar para aprender.")
    with d3:
        categorias = ", ".join(f"<b>{c}</b>" for c in DADOS[LABEL].unique())
        cartao("🏷️", "Saída (label)",
               f"Coluna <b>{LABEL}</b>: tem poucas categorias que se repetem ({categorias}), "
               "então é a resposta que o modelo deve aprender a prever.")

    st.write("")

    def pintar(coluna):
        if coluna.name in st.session_state.features:
            return ["background-color: #ddeef7; color:#12506b; font-weight:600"] * len(coluna)
        if coluna.name == LABEL:
            return ["background-color: #fdf3d7; color:#8a6a08; font-weight:600"] * len(coluna)
        return ["color:#8aa0aa"] * len(coluna)

    st.dataframe(DADOS.style.apply(pintar, axis=0), width='stretch', hide_index=True)
    st.caption("Azul: features selecionadas | Amarelo: label | Cinza: colunas ignoradas pelo modelo")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Escolha as features")
        explica("Experimente! Você decide quais dessas informações o modelo poderá usar para aprender. Marque pelo menos uma.")
        selecionadas = []
        for f in FEATURES_DISPONIVEIS:
            if st.checkbox(f, value=f in st.session_state.features, key=f"chk_{f}"):
                selecionadas.append(f)
        if selecionadas != st.session_state.features:
            st.session_state.features = selecionadas
            st.session_state.treinado = False
            st.rerun()

    with c2:
        st.subheader("Como os dados se distribuem")
        if len(st.session_state.features) >= 1:
            eixo_x = st.session_state.features[0]
            eixo_y = st.session_state.features[1] if len(st.session_state.features) > 1 else eixo_x
            fig, ax = plt.subplots(figsize=(5.5, 4))
            paleta = ["#2fa07a", "#d15b5b", "#3c7dd9", "#c98a2d", "#8256c9"]
            nomes = nomes_das_linhas()
            for i, (situacao, grupo) in enumerate(DADOS.groupby(LABEL)):
                ax.scatter(grupo[eixo_x], grupo[eixo_y], s=140, alpha=.85,
                           color=paleta[i % len(paleta)], label=str(situacao),
                           edgecolors="white", linewidths=1.5)
            for idx, linha in DADOS.iterrows():
                ax.annotate(str(nomes.loc[idx]), (linha[eixo_x], linha[eixo_y]),
                            textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8, color="#456")
            ax.set_xlabel(eixo_x)
            ax.set_ylabel(eixo_y)
            ax.legend()
            ax.grid(alpha=.25)
            fig.patch.set_facecolor("#ffffff")
            st.pyplot(fig, width='stretch')
            plt.close(fig)
            explica("Repare: cada cor é uma categoria do label. Quando as cores formam <b>grupos separados</b> no gráfico, existe um padrão claro para o modelo descobrir!")
        else:
            st.info("Selecione pelo menos uma feature para ver o gráfico.")

    pode = len(st.session_state.features) >= 1
    botoes_navegacao(pode_avancar=pode, aviso="Selecione pelo menos uma feature para continuar.")

# ============================================================================
# ETAPA 3 - DIVISAO TREINO / TESTE
# ============================================================================
elif etapa == 2:
    st.header("✂️ Etapa 3: Dividindo os dados")
    explica(
        "Imagine estudar para uma prova usando as mesmas questões que cairão nela. Você acertaria tudo, "
        "mas isso não provaria que aprendeu de verdade. Com a IA é igual! Por isso separamos os dados em dois grupos:<br><br>"
        "🏋️ <b>Treinamento:</b> exemplos que o modelo usa para aprender.<br>"
        "📝 <b>Teste:</b> exemplos escondidos do modelo, usados depois para verificar se ele realmente aprendeu."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        pct = st.slider("Porcentagem dos dados usada para TREINO", 50, 90, st.session_state.pct_treino, step=10,
                        help="O restante vai para o grupo de teste.")
    with c2:
        if st.button("🎲 Sortear novamente", width='stretch'):
            st.session_state.semente += 1
            st.session_state.treinado = False

    if pct != st.session_state.pct_treino:
        st.session_state.pct_treino = pct
        st.session_state.treinado = False

    X_tr, X_te, y_tr, y_te, nomes_tr, nomes_te = dividir_dados()

    n_tr, n_te = len(nomes_tr), len(nomes_te)
    st.progress(pct / 100, text=f"Treino: {pct}% ({n_tr} alunos)  |  Teste: {100 - pct}% ({n_te} alunos)")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader(f"🏋️ Grupo de treino ({n_tr})")
        chips = "".join(f"<span class='aluno-chip treino' style='animation-delay:{i*0.12}s'>🏋️ {n}</span>" for i, n in enumerate(nomes_tr))
        st.markdown(chips, unsafe_allow_html=True)
        explica(f"Estes alunos serão mostrados ao modelo <b>com a resposta</b> (a coluna {LABEL}).")
    with c4:
        st.subheader(f"📝 Grupo de teste ({n_te})")
        chips = "".join(f"<span class='aluno-chip teste' style='animation-delay:{i*0.12}s'>📝 {n}</span>" for i, n in enumerate(nomes_te))
        st.markdown(chips, unsafe_allow_html=True)
        explica("Estes alunos ficam <b>escondidos</b> durante o treino. Serão a prova final do modelo!")

    botoes_navegacao(pode_avancar=n_te >= 1, aviso="Deixe pelo menos 1 aluno no grupo de teste.")

# ============================================================================
# ETAPA 4 - ESCOLHA DO ALGORITMO E TREINAMENTO
# ============================================================================
elif etapa == 3:
    st.header("🌳 Etapa 4: Escolhendo o algoritmo e treinando")

    algoritmo = st.selectbox("Algoritmo de aprendizado", ["Árvore de Decisão"], index=0)

    c1, c2 = st.columns(2)
    with c1:
        cartao("🌳", "Como pensa uma Árvore de Decisão?",
               "Ela aprende fazendo <b>perguntas de sim ou não</b> sobre os dados, como: "
               "<i>'A nota é maior que 56?'</i>. Cada resposta leva a uma nova pergunta, "
               "até chegar a uma conclusão. É como um jogo de 'Cara a Cara' com os dados!")
    with c2:
        cartao("🔍", "Como ela escolhe as perguntas?",
               "A cada passo, o algoritmo testa várias perguntas possíveis e escolhe a que <b>melhor separa</b> "
               "as categorias do label. As perguntas mais importantes ficam no topo da árvore.")

    profundidade = st.slider("Profundidade máxima da árvore (quantos níveis de perguntas)", 1, 4, 3,
                             help="Árvores mais profundas fazem mais perguntas e criam regras mais detalhadas.")

    st.write("")
    if st.button("🚀 Treinar o modelo agora!", type="primary", width='stretch'):
        X_tr, X_te, y_tr, y_te, nomes_tr, nomes_te = dividir_dados()
        area = st.empty()
        status = st.empty()

        for nivel in range(1, profundidade + 1):
            status.info(f"🧠 O modelo está aprendendo... construindo o nível {nivel} de perguntas.")
            modelo_parcial = DecisionTreeClassifier(max_depth=nivel, random_state=0)
            modelo_parcial.fit(X_tr, y_tr)
            with area.container():
                desenhar_arvore(modelo_parcial, st.session_state.features,
                                titulo=f"Aprendendo... nível {nivel} de {profundidade}")
            time.sleep(1.2)

        modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=0)
        modelo.fit(X_tr, y_tr)
        st.session_state.modelo = modelo
        st.session_state.treinado = True
        status.success("✅ Treinamento concluído! O modelo descobriu suas regras.")
        st.balloons()

    if st.session_state.treinado and st.session_state.modelo is not None:
        st.subheader("A árvore que o modelo aprendeu")
        desenhar_arvore(st.session_state.modelo, st.session_state.features, "Modelo final treinado")
        explica(
            "📖 <b>Como ler a árvore:</b> comece pelo topo. Cada caixa faz uma pergunta sobre um aluno. "
            "Se a resposta for <b>sim</b> (True), siga para a esquerda; se for <b>não</b> (False), para a direita. "
            "As caixas finais (folhas) mostram a decisão. "
            "O campo <i>samples</i> indica quantos alunos do treino chegaram até ali."
        )
        imp = pd.Series(st.session_state.modelo.feature_importances_, index=st.session_state.features).sort_values()
        st.subheader("Quais informações pesaram mais?")
        fig, ax = plt.subplots(figsize=(7, 2.2))
        imp.plot.barh(ax=ax, color="#2b8fa3")
        ax.set_xlabel("Importância no modelo")
        fig.patch.set_facecolor("#ffffff")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    botoes_navegacao(pode_avancar=st.session_state.treinado,
                     aviso="Treine o modelo antes de avançar. Clique no botão acima!")

# ============================================================================
# ETAPA 5 - AVALIACAO
# ============================================================================
elif etapa == 4:
    st.header("📝 Etapa 5: Hora da prova! Avaliando o modelo")

    if not st.session_state.treinado:
        st.warning("O modelo ainda não foi treinado. Volte para a etapa 4.")
        botoes_navegacao(pode_avancar=False, aviso="")
    else:
        X_tr, X_te, y_tr, y_te, nomes_tr, nomes_te = dividir_dados()
        modelo = st.session_state.modelo

        explica(
            "Agora chegou a hora da verdade. Vamos mostrar ao modelo os alunos do grupo de <b>teste</b>, "
            "que ele <b>nunca viu</b>, e comparar as previsões dele com a realidade."
        )

        previsoes = modelo.predict(X_te)
        acuracia = accuracy_score(y_te, previsoes)

        resultado = pd.DataFrame({
            "Aluno": list(nomes_te),
            f"{LABEL} real": list(y_te),
            "Previsão do modelo": list(previsoes),
        })
        resultado["Resultado"] = ["✅ Acertou" if r == p else "❌ Errou"
                                  for r, p in zip(resultado[f"{LABEL} real"], resultado["Previsão do modelo"])]

        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(resultado, width='stretch', hide_index=True)
        with c2:
            st.metric("🎯 Acurácia no teste", f"{acuracia:.0%}",
                      help="Porcentagem de previsões corretas nos alunos que o modelo nunca viu.")
            acertos = int((resultado["Resultado"] == "✅ Acertou").sum())
            st.metric("Acertos", f"{acertos} de {len(resultado)}")

        if acuracia == 1:
            explica("🌟 <b>Perfeito!</b> O modelo acertou todos os alunos do teste. Isso indica que ele realmente aprendeu o padrão, e não apenas decorou os exemplos do treino.")
        elif acuracia >= 0.5:
            explica("🙂 O modelo acertou parte dos casos. Com poucos dados, isso é normal. Experimente voltar e mudar as features, a divisão dos dados ou a profundidade da árvore para ver se melhora!")
        else:
            explica("😅 O modelo errou bastante. Volte às etapas anteriores e experimente outras combinações. Errar e ajustar também faz parte do Machine Learning!")

        explica(
            "💡 <b>Conceito importante:</b> avaliar o modelo em dados que ele nunca viu chama-se "
            "<b>generalização</b>. Um bom modelo não decora, ele generaliza."
        )
        botoes_navegacao()

# ============================================================================
# ETAPA 6 - PLAYGROUND DE PREVISAO
# ============================================================================
elif etapa == 5:
    st.header("🔮 Etapa 6: Crie um aluno e veja a previsão")

    if not st.session_state.treinado:
        st.warning("O modelo ainda não foi treinado. Volte para a etapa 4.")
        botoes_navegacao(pode_avancar=False, aviso="")
    else:
        modelo = st.session_state.modelo
        explica(
            "O modelo está treinado e pronto para trabalhar! Invente um aluno novo movendo os controles abaixo "
            "e veja, em tempo real, o caminho que a árvore percorre até a decisão. "
            "Os limites de cada controle foram calculados a partir dos valores do próprio CSV."
        )

        valores = {}
        cols = st.columns(len(st.session_state.features))
        for col, f in zip(cols, st.session_state.features):
            minimo, maximo, padrao = limites_do_slider(f)
            with col:
                valores[f] = st.slider(f, minimo, maximo, padrao)

        entrada = pd.DataFrame([valores])[st.session_state.features]
        previsao = modelo.predict(entrada)[0]
        probabilidades = modelo.predict_proba(entrada)[0]
        confianca = probabilidades.max()

        positiva = classe_positiva()
        if positiva is None:
            cor = "neutro"
            emoji = "🔮"
        elif previsao == positiva:
            cor, emoji = "aprovado", "🎉"
        else:
            cor, emoji = "reprovado", "📚"
        st.markdown(
            f"<h3>Previsão do modelo: <span class='aluno-chip {cor}' style='font-size:20px'>{emoji} {previsao}</span> "
            f"<small style='color:#6b8894'>(confiança de {confianca:.0%})</small></h3>",
            unsafe_allow_html=True,
        )

        st.subheader("🧭 O caminho da decisão, passo a passo")
        arvore = modelo.tree_
        caminho = modelo.decision_path(entrada).indices
        passos = []
        for no in caminho:
            if arvore.children_left[no] == -1:
                continue
            f_idx = arvore.feature[no]
            nome_f = st.session_state.features[f_idx]
            limiar = arvore.threshold[no]
            valor = valores[nome_f]
            if valor <= limiar:
                passos.append(f"❓ <b>{nome_f}</b> = {valor} é menor ou igual a {limiar:.1f}? <b>Sim</b> ➜ segue pela esquerda.")
            else:
                passos.append(f"❓ <b>{nome_f}</b> = {valor} é menor ou igual a {limiar:.1f}? <b>Não</b> ➜ segue pela direita.")
        passos.append(f"🏁 Chegou a uma folha da árvore: decisão final <b>{previsao}</b>.")
        explica("<br>".join(passos))

        with st.expander("Ver a árvore completa novamente"):
            desenhar_arvore(modelo, st.session_state.features, "Modelo treinado")

        st.divider()
        explica(
            "🎓 <b>Parabéns, você concluiu o laboratório!</b> Você acompanhou o ciclo completo do "
            "aprendizado supervisionado: entender os dados ➜ separar treino e teste ➜ treinar ➜ avaliar ➜ prever. "
            "Esse mesmo ciclo é usado em sistemas reais de IA, apenas com muito mais dados. "
            "Volte às etapas anteriores, experimente outras combinações ou envie seu próprio CSV pela barra lateral!"
        )
        botoes_navegacao()
