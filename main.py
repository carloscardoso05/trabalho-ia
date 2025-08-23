import random as rd

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class PredictDados(BaseModel):
    nome: str
    # Adicionar mais campos depois


cargos = [
    "DBA/Administrador de Banco de Dados",
    "Desenvolvedor/ Engenheiro de Software/ Analista de Sistemas",
    "Cientista de Dados/Data Scientist",
    "Professor",
    "Analista de BI/BI Analyst",
    "Analista de Inteligência de Mercado/Market Intelligence",
    "Analista de Negócios/Business Analyst",
    "Engenheiro de Dados/Arquiteto de Dados/Data Engineer/Data Architect",
    "Analista de Dados/Data Analyst",
    "Product Manager/ Product Owner (PM/APM/DPM/GPM/PO)",
    "Outra Opção",
    "Analista de Suporte/Analista Técnico",
    "Engenheiro de Machine Learning/ML Engineer",
    "Analytics Engineer",
    "Analista de Marketing",
    "Outras Engenharias (não inclui dev)",
    "Estatístico",
    "Economista",
]


def predict_cargo(dados: PredictDados):
    """
    A implementação real será consultando o modelo treinado.

    O parâmetro "dados" será um objeto contentod os dados do
    usuário que foram inseridos no formulátio (nome, data de nascimento, linguagem preferida, etc.)
    """
    return rd.choice(cargos)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, nome: str = Form()):
    cargo = predict_cargo(PredictDados(nome=nome))

    mensagem = f"{nome}, seu cargo recomendado é: {cargo}!"

    return templates.TemplateResponse(
        request=request, name="result.html", context={"mensagem": mensagem}
    )
