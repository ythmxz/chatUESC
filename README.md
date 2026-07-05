# ChatUESC

Chatbot de perguntas e respostas sobre a UESC com pipeline local de:

1. Coleta de páginas do domínio uesc.br.
2. Geração de chunks de contexto para recuperação.
3. Busca semântica com TF-IDF.
4. Resposta com Gemini usando somente o contexto recuperado.

## Arquitetura

- crawler.py: rastreia páginas da UESC e salva em data/pages.json.
- build_index.py: transforma páginas em chunks e salva em chunks.json.
- chatbot.py: recupera contexto relevante e gera respostas em streaming.
- main.py: integra o fluxo completo em um único comando.

## Pré-requisitos

- Python 3.10+
- Dependências do projeto
- Chave de API do Google AI Studio: https://aistudio.google.com/api-keys

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install requests beautifulsoup4 scikit-learn scipy numpy google-genai
```

## Configuração da API

Opção recomendada (variável de ambiente):

```powershell
$env:GOOGLE_API_KEY="SUA_CHAVE_AQUI"
```

Alternativa:

- Preencher a constante API_KEY em chatbot.py.

## Execução

Fluxo completo com um comando:

```powershell
python .\main.py
```

Opções úteis:

```powershell
python .\main.py --skip-crawl
python .\main.py --skip-index
```

Durante o chat:

- Digite perguntas normalmente.
- Digite 0 para encerrar.

## Execução modular (opcional)

```powershell
python .\crawler.py
python .\build_index.py
python .\chatbot.py
```

## Estrutura de dados gerados

- data/pages.json: lista de páginas coletadas (url, title, text).
- chunks.json: lista de chunks para recuperação de contexto.

## A Fazer

- Implementar recuperação baseada em TF-IDF com `sentence-transformers` usando modelo local `all-MiniLM-L6-v2` (não usa embeddings densos).
- Implementar cache de respostas ou detecção de resposta já contida no contexto final.
