# ChatUESC

Projeto de chatbot para consultar informações sobre a UESC.

## Execução

### 1. Coletar páginas

```powershell
python .\crawler.py
```

### 2. Extrair contexto

```powershell
python .\build_index.py
```

### 3. Iniciar chatbot

```powershell
python .\chatbot.py
```

## A Fazer

- Implementar busca com `sentence-transformers` usando modelo local `all-MiniLM-L6-v2`.
- Implementar detecção de respostas contidas nos contextos para economizar tokens.
