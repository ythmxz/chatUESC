from google.genai import Client
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from json import load
from os import getenv
from numpy import dtype, float64, int64, ndarray
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Iterator


API_KEY: str = ""


def load_chunks(chunks_path: str = "chunks.json") -> list[dict[str, str]]:
    """Carrega os chunks de contexto que alimentam a recuperação de informação."""
    with open(chunks_path, encoding="utf8") as f:
        return load(f)


def build_vector_index(chunks: list[dict[str, str]]) -> tuple[TfidfVectorizer, spmatrix]:
    """Cria vetorizador TF-IDF e matriz de documentos para busca por similaridade."""
    documents: list[str] = [chunk["text"] for chunk in chunks]
    vectorizer: TfidfVectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix: spmatrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def search_context(
    question: str,
    chunks: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    matrix: spmatrix,
    k: int = 15,
) -> list[dict[str, str]]:
    """Recupera os chunks mais relevantes para a pergunta evitando duplicação por URL."""
    question_vector: spmatrix = vectorizer.transform([question])
    similarities: ndarray[Any, dtype[float64]] = cosine_similarity(question_vector, matrix)[0]
    indexes: ndarray[Any, dtype[int64]] = similarities.argsort()[-k:][::-1]

    results: list[dict[str, str]] = []
    used_urls: set[str] = set()

    for index in indexes:
        url: str = chunks[index]["url"]

        if url in used_urls:
            continue

        used_urls.add(url)
        results.append(chunks[index])

        if len(results) >= k:
            break

    return results


def answer(
    question: str,
    client: Client,
    chunks: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    matrix: spmatrix,
    model: str = "gemini-3.5-flash",
) -> Iterator[GenerateContentResponse]:
    """Gera resposta em streaming usando Gemini com contexto recuperado localmente."""
    context: list[dict[str, str]] = search_context(question, chunks, vectorizer, matrix)
    context_text = "\n\n".join(item["text"] for item in context)

    prompt: str = f"""
    CONTEXTO:
    {context_text}

    INSTRUÇÕES:

    - Use somente o contexto.
    - Se houver uma lista completa no contexto, reproduza a lista completa.
    - Não resuma listas.
    - Não omita itens.

    PERGUNTA:
    {question}
    """

    response = client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction="""
            Você é um assistente da UESC.

            Priorize o contexto recebido.

            Se a resposta não estiver no contexto,
            informe que não encontrou a informação.

            Não informe que está buscando ou que a resposta é baseada em um contexto recebido.
            """
        ))

    return response


def run_chat(api_key: str, chunks_path: str = "chunks.json") -> None:
    """Inicia loop interativo do chatbot até o usuário informar 0."""
    chunks = load_chunks(chunks_path)
    vectorizer, matrix = build_vector_index(chunks)
    client = Client(api_key=api_key)

    while True:
        question = input("(Você): ").strip()
        print()

        if question == "0":
            break

        response = answer(question, client, chunks, vectorizer, matrix)

        print("(IA): ", end=" ")
        for chunk in response:
            print(chunk.text, end=" ")

        print(end="\n\n")


def main() -> None:
    """Resolve API key e inicia o chatbot no modo interativo."""
    api_key = getenv("GOOGLE_API_KEY") or API_KEY
    if not api_key:
        raise ValueError(
            "Configure GOOGLE_API_KEY no ambiente ou preencha API_KEY em chatbot.py"
        )

    run_chat(api_key=api_key)


if __name__ == "__main__":
    main()
