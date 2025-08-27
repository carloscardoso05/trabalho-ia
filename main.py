import pickle
import random as rd
from typing import List

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

with open("modelos/modelo-lr.pkl", "rb") as file:
    modelo = pickle.load(file)

colunas = [
    "idade",
    "qtd_bancos",
    "nivel_ensino_Doutorado ou Phd",
    "nivel_ensino_Estudante de Graduação",
    "nivel_ensino_Graduação/Bacharelado",
    "nivel_ensino_Mestrado",
    "nivel_ensino_Pós-graduação",
    "formacao_Ciências Biológicas/ Farmácia/ Medicina/ Área da Saúde",
    "formacao_Ciências Sociais",
    "formacao_Computação / Engenharia de Software / Sistemas de Informação/ TI",
    "formacao_Economia/ Administração / Contabilidade / Finanças/ Negócios",
    "formacao_Estatística/ Matemática / Matemática Computacional/ Ciências Atuariais",
    "formacao_Marketing / Publicidade / Comunicação / Jornalismo",
    "formacao_Outra opção",
    "formacao_Outras Engenharias",
    "formacao_Química / Física",
    "tempo_experiencia_dados_Mais de 10 anos",
    "tempo_experiencia_dados_Menos de 1 ano",
    "tempo_experiencia_dados_Não tenho experiência na área de dados",
    "tempo_experiencia_dados_de 1 a 2 anos",
    "tempo_experiencia_dados_de 3 a 4 anos",
    "tempo_experiencia_dados_de 4 a 6 anos",
    "tempo_experiencia_dados_de 7 a 10 anos",
    "linguagens_preferidas_aql",
    "linguagens_preferidas_c/c++/c#",
    "linguagens_preferidas_clojure",
    "linguagens_preferidas_elixir",
    "linguagens_preferidas_julia",
    "linguagens_preferidas_nenhuma",
    "linguagens_preferidas_outros",
    "linguagens_preferidas_python",
    "linguagens_preferidas_r",
    "linguagens_preferidas_rust",
    "linguagens_preferidas_scala",
    "linguagens_preferidas_spark",
    "linguagens_preferidas_sql",
    "bancos_de_dados_amazon",
    "bancos_de_dados_azure",
    "bancos_de_dados_bigquery",
    "bancos_de_dados_cassandra",
    "bancos_de_dados_databricks",
    "bancos_de_dados_datomic",
    "bancos_de_dados_db2",
    "bancos_de_dados_dynamodb",
    "bancos_de_dados_elasticsearch",
    "bancos_de_dados_excel",
    "bancos_de_dados_firebase",
    "bancos_de_dados_firebird",
    "bancos_de_dados_hbase",
    "bancos_de_dados_hive",
    "bancos_de_dados_ibm",
    "bancos_de_dados_interno",
    "bancos_de_dados_mariadb",
    "bancos_de_dados_microsoft",
    "bancos_de_dados_mongoDB",
    "bancos_de_dados_nenhum",
    "bancos_de_dados_neo4j",
    "bancos_de_dados_oracle",
    "bancos_de_dados_outros",
    "bancos_de_dados_presto",
    "bancos_de_dados_redis",
    "bancos_de_dados_s3",
    "bancos_de_dados_sap",
    "bancos_de_dados_sharepoint",
    "bancos_de_dados_snowflake",
    "bancos_de_dados_splunk",
    "bancos_de_dados_sql",
    "bancos_de_dados_sybase",
    "bancos_de_dados_teradata",
    "cloud_preferida_Amazon Web Services (AWS)",
    "cloud_preferida_Azure (Microsoft)",
    "cloud_preferida_Google Cloud (GCP)",
    "cloud_preferida_Não sei opinar",
    "cloud_preferida_Outra Cloud",
]


class PredictDados(BaseModel):
    nome: str
    idade: int
    nivel_ensino: str
    formacao: str
    tempo_experiencia_dados: str
    linguagens_preferidas: str
    bancos_de_dados: List[str]
    cloud_preferida: str


cargos = [
    "Analista de BI/BI Analyst",
    "Analista de Dados/Data Analyst",
    "Cientista de Dados/Data Scientist",
    "Engenheiro de Dados/Arquiteto de Dados/Data Engineer/Data Architect",
]


def predict_cargo(dados: PredictDados):
    """
    Usa o modelo treinado para predizer o cargo baseado nos dados fornecidos.
    """
    # Criar vetor de features zerado com o tamanho das colunas esperadas
    features = [0] * len(colunas)

    # Preencher as features baseadas nos dados fornecidos
    # Idade
    if "idade" in colunas:
        features[colunas.index("idade")] = dados.idade

    # Quantidade de bancos (calculada automaticamente baseada nas seleções)
    qtd_bancos_calculada = len(dados.bancos_de_dados)
    if "qtd_bancos" in colunas:
        features[colunas.index("qtd_bancos")] = qtd_bancos_calculada

    # Nível de ensino (one-hot encoding)
    nivel_col = f"nivel_ensino_{dados.nivel_ensino}"
    if nivel_col in colunas:
        features[colunas.index(nivel_col)] = 1

    # Formação (one-hot encoding)
    formacao_col = f"formacao_{dados.formacao}"
    if formacao_col in colunas:
        features[colunas.index(formacao_col)] = 1

    # Tempo de experiência (one-hot encoding)
    exp_col = f"tempo_experiencia_dados_{dados.tempo_experiencia_dados}"
    if exp_col in colunas:
        features[colunas.index(exp_col)] = 1

    # Linguagens preferidas (one-hot encoding)
    lang_col = f"linguagens_preferidas_{dados.linguagens_preferidas}"
    if lang_col in colunas:
        features[colunas.index(lang_col)] = 1

    # Bancos de dados (one-hot encoding para múltiplos valores)
    if dados.bancos_de_dados:  # Se há bancos selecionados
        for banco in dados.bancos_de_dados:
            bd_col = f"bancos_de_dados_{banco}"
            if bd_col in colunas:
                features[colunas.index(bd_col)] = 1
    else:  # Se nenhum banco foi selecionado, marcar como "nenhum"
        bd_col = "bancos_de_dados_nenhum"
        if bd_col in colunas:
            features[colunas.index(bd_col)] = 1

    # Cloud preferida (one-hot encoding)
    cloud_col = f"cloud_preferida_{dados.cloud_preferida}"
    if cloud_col in colunas:
        features[colunas.index(cloud_col)] = 1

    # Fazer a predição
    try:
        prediction = modelo.predict([features])[0]
        return prediction
    except Exception as e:
        print(f"Erro na predição: {e}")
        # Fallback para escolha aleatória
        return rd.choice(cargos)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Criar listas para popular os selects do formulário
    # Nível de ensino: do menos ao mais instruído
    nivel_ensino_options = [
        "Estudante de Graduação",
        "Graduação/Bacharelado",
        "Pós-graduação",
        "Mestrado",
        "Doutorado ou Phd",
    ]

    # Formação: ordem alfabética, com "outras" no final
    formacao_options = [
        "Ciências Biológicas/ Farmácia/ Medicina/ Área da Saúde",
        "Ciências Sociais",
        "Computação / Engenharia de Software / Sistemas de Informação/ TI",
        "Economia/ Administração / Contabilidade / Finanças/ Negócios",
        "Estatística/ Matemática / Matemática Computacional/ Ciências Atuariais",
        "Marketing / Publicidade / Comunicação / Jornalismo",
        "Química / Física",
        "Outras Engenharias",
        "Outra opção",
    ]

    # Tempo de experiência: do menos ao mais experiente
    tempo_experiencia_options = [
        "Não tenho experiência na área de dados",
        "Menos de 1 ano",
        "de 1 a 2 anos",
        "de 3 a 4 anos",
        "de 4 a 6 anos",
        "de 7 a 10 anos",
        "Mais de 10 anos",
    ]

    # Linguagens: populares primeiro, depois alfabética, com genéricas no final
    linguagens_options = [
        "python",
        "sql",
        "r",
        "c/c++/c#",
        "scala",
        "julia",
        "rust",
        "spark",
        "aql",
        "clojure",
        "elixir",
        "nenhuma",
        "outros",
    ]

    # Bancos de dados: populares primeiro, depois alfabética, com genéricos no final
    bancos_options = [
        "sql",
        "oracle",
        "mongoDB",
        "redis",
        "elasticsearch",
        "snowflake",
        "bigquery",
        "amazon",
        "azure",
        "s3",
        "databricks",
        "cassandra",
        "db2",
        "dynamodb",
        "excel",
        "firebase",
        "firebird",
        "hbase",
        "hive",
        "ibm",
        "mariadb",
        "microsoft",
        "neo4j",
        "presto",
        "sap",
        "sharepoint",
        "splunk",
        "sybase",
        "teradata",
        "datomic",
        "interno",
        "outros",
    ]

    # Cloud: principais providers primeiro, depois genéricas
    cloud_options = [
        "Amazon Web Services (AWS)",
        "Azure (Microsoft)",
        "Google Cloud (GCP)",
        "Outra Cloud",
        "Não sei opinar",
    ]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "nivel_ensino_options": nivel_ensino_options,
            "formacao_options": formacao_options,
            "tempo_experiencia_options": tempo_experiencia_options,
            "linguagens_options": linguagens_options,
            "bancos_options": bancos_options,
            "cloud_options": cloud_options,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    nome: str = Form(),
    idade: int = Form(),
    nivel_ensino: str = Form(),
    formacao: str = Form(),
    tempo_experiencia_dados: str = Form(),
    linguagens_preferidas: str = Form(),
    bancos_de_dados: List[str] = Form(),
    cloud_preferida: str = Form(),
):
    dados = PredictDados(
        nome=nome,
        idade=idade,
        nivel_ensino=nivel_ensino,
        formacao=formacao,
        tempo_experiencia_dados=tempo_experiencia_dados,
        linguagens_preferidas=linguagens_preferidas,
        bancos_de_dados=bancos_de_dados,
        cloud_preferida=cloud_preferida,
    )

    cargo = predict_cargo(dados)

    mensagem = f"{nome}, seu cargo recomendado é: {cargos[cargo]}!"

    return templates.TemplateResponse(
        request=request, name="result.html", context={"mensagem": mensagem}
    )
