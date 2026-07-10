# ChatUESC

Chatbot de perguntas e respostas sobre a Universidade Estadual de Santa Cruz (UESC) utilizando Recuperação Aumentada por Geração (RAG).

O sistema coleta páginas do site da UESC, gera uma base local de conhecimento, recupera os trechos mais relevantes utilizando TF-IDF e utiliza o Gemini para gerar respostas contextualizadas.

## Arquitetura

```text
crawler.py
    ↓
data/pages.json
    ↓
build_index.py
    ↓
data/chunks.json
    ↓
chatbot.py
    ↓
Gemini
```

### Componentes

- **crawler.py**: realiza a coleta de páginas do domínio `uesc.br`.
- **build_index.py**: divide as páginas em chunks para recuperação de contexto.
- **chatbot.py**: executa a recuperação de contexto, mantém a conversa e gera respostas com o Gemini.
- **main.py**: ponto de entrada da aplicação.

## Funcionamento

1. O crawler percorre o site da UESC utilizando Busca em Largura (BFS).
2. As páginas coletadas são armazenadas em `data/pages.json`.
3. O indexador divide os textos em chunks com sobreposição.
4. Os chunks são armazenados em `data/chunks.json`.
5. A pergunta do usuário é transformada em um vetor TF-IDF.
6. Os chunks mais relevantes são recuperados utilizando similaridade do cosseno.
7. O contexto recuperado é enviado ao Gemini.
8. O Gemini gera a resposta final mantendo o histórico da conversa.

## Tecnologias Utilizadas

- Python 3.10+
- Google Gemini API
- scikit-learn
- NumPy
- SciPy
- Requests
- Beautiful Soup 4

## Instalação

Crie e ative um ambiente virtual:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

## Configuração

Defina a chave da API Gemini através da variável de ambiente:

### Windows (PowerShell)

```powershell
$env:GEMINI_API_KEY="SUA_CHAVE"
```

### Linux

```bash
export GEMINI_API_KEY="SUA_CHAVE"
```

Opcionalmente, também é possível definir o modelo:

```bash
export GEMINI_MODEL="gemini-3-flash-preview"
```

## Estrutura do Projeto

```text
chatUESC/
├── data/
│   ├── chunks.json
│   └── pages.json
├── build_index.py
├── chatbot.py
├── crawler.py
├── main.py
├── README.md
└── requirements.txt
```

## Uso

### Executar apenas o chatbot

```bash
python main.py
```

### Atualizar páginas e chunks

```bash
python main.py -u
```

ou

```bash
python main.py --update
```

### Executar apenas o crawler

```bash
python main.py -c
```

ou

```bash
python main.py --crawl
```

### Reconstruir apenas os chunks

```bash
python main.py -b
```

ou

```bash
python main.py --build
```

## Execução Modular

Também é possível executar cada etapa separadamente.

### Coleta de páginas

```bash
python crawler.py
```

### Geração dos chunks

```bash
python build_index.py
```

### Chatbot

```bash
python chatbot.py
```

## Conversação

Durante a execução do chatbot:

- Faça perguntas normalmente.
- Digite `0` para encerrar a aplicação.

## Recuperação de Contexto

O sistema utiliza:

- TF-IDF com unigramas e bigramas (`ngram_range=(1, 2)`).
- Similaridade do cosseno.
- Deduplicação de URLs.
- Seleção dos chunks mais relevantes.
- Fallback para conhecimento geral do modelo quando não houver contexto suficiente.

## Crawler

O crawler possui as seguintes características:

- Busca em largura (BFS).
- Priorização de páginas institucionais.
- Menor prioridade para notícias e eventos.
- Reutilização de conexões HTTP com `requests.Session`.
- Retry automático para falhas temporárias.
- Remoção de parâmetros de rastreamento.
- Filtragem de arquivos binários (PDF, imagens, planilhas, apresentações etc.).

## Arquivos Gerados

### data/pages.json

Contém as páginas coletadas:

```json
{
  "url": "...",
  "title": "...",
  "text": "..."
}
```

### data/chunks.json

Contém os chunks utilizados na recuperação:

```json
{
  "url": "...",
  "title": "...",
  "text": "..."
}
```

## Limitações

- A qualidade das respostas depende do conteúdo coletado pelo crawler.
- O sistema utiliza TF-IDF, não embeddings semânticos.
- O conhecimento da base precisa ser atualizado manualmente através do crawler.
