from json import load
from os import getenv
from typing import Any, cast

from google.genai import Client
from google.genai.chats import Chat
from google.genai.types import GenerateContentConfig
from numpy import dtype, float64, int64, ndarray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL: str = getenv("GEMINI_MODEL", "gemini-2.5-flash")
TOP_K: int = 10
SIMILARITY_THRESHOLD: float = 0.10
SYSTEM_INSTRUCTION: str = """
Você é um assistente da Universidade Estadual de Santa Cruz (UESC).

Responda em português do Brasil.

Mantenha o contexto da conversa atual.

Não invente informações.

Quando não souber uma resposta, deixe isso claro.
"""


def load_chunks(path: str) -> list[dict[str, str]]:
    """
    Carrega os chunks indexados.

    Args:
        path: Caminho do arquivo JSON contendo os chunks.

    Returns:
        Lista de chunks.
    """
    with open(path, encoding="utf8") as file:
        return load(file)


def build_vectorizer(documents: list[str]) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Constrói o modelo TF-IDF dos documentos.

    Args:
        documents: Textos utilizados na construção do índice.

    Returns:
        Tupla contendo o vetorizador treinado e a matriz TF-IDF.

    Raises:
        ValueError: Caso a lista de documentos esteja vazia.
    """
    if not documents:
        raise ValueError("Não é possível construir o índice sem documentos.")

    vectorizer: TfidfVectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))

    matrix: csr_matrix = cast(csr_matrix, vectorizer.fit_transform(documents))

    return vectorizer, matrix


def create_chat(client: Client) -> Chat:
    """
    Cria uma sessão de chat persistente.

    Args:
        client: Cliente Gemini.

    Returns:
        Sessão de chat.
    """
    return client.chats.create(
        model=MODEL, config=GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
    )


def search_context(
    question: str,
    chunks: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    k: int = TOP_K,
) -> tuple[list[dict[str, str]], ndarray[Any, dtype[float64]]]:
    """
    Recupera os chunks mais relevantes para uma pergunta.

    Chunks da mesma URL são deduplicados para evitar
    desperdício de contexto.

    Args:
        question: Pergunta do usuário.
        chunks: Lista de chunks indexados.
        vectorizer: Vetorizador TF-IDF.
        matrix: Matriz TF-IDF.
        k: Quantidade máxima de chunks.

    Returns:
        Tupla contendo:
        - Lista de chunks relevantes.
        - Vetor de similaridades.
    """
    question_vector: csr_matrix = cast(csr_matrix, vectorizer.transform([question]))

    similarities: ndarray[Any, dtype[float64]] = cosine_similarity(
        question_vector, matrix
    )[0]

    indexes: ndarray[Any, dtype[int64]] = similarities.argsort()[::-1]

    selected_chunks: list[dict[str, str]] = []
    selected_scores: list[float] = []

    used_urls: set[str] = set()

    for index in indexes:
        score: float = float(similarities[index])

        if score < SIMILARITY_THRESHOLD:
            break

        url: str = chunks[index]["url"]

        if url in used_urls:
            continue

        used_urls.add(url)
        selected_chunks.append(chunks[index])
        selected_scores.append(score)

        if len(selected_chunks) >= k:
            break

    return (
        selected_chunks,
        ndarray(shape=(len(selected_scores),), dtype=float64, buffer=None)
        if False
        else __import__("numpy").array(selected_scores, dtype=float64),
    )


def build_prompt(question: str, context: list[dict[str, str]]) -> str:
    """
    Constrói o prompt enviado ao Gemini.

    Args:
        question: Pergunta do usuário.
        context: Chunks recuperados.

    Returns:
        Prompt formatado.
    """
    if len(context) == 0:
        return f"""
        Pergunta:

        {question}
        """

    context_text: str = "\n\n".join(chunk["text"] for chunk in context)

    return f"""
    CONTEXTO:

    {context_text}

    PERGUNTA:

    {question}

    Utilize prioritariamente o contexto acima.
    Caso necessário, complemente a resposta com conhecimento geral.
    """


def answer(
    question: str,
    chat: Chat,
    chunks: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
) -> str:
    """
    Responde uma pergunta do usuário.

    Args:
        question: Pergunta do usuário.
        chat: Sessão de chat Gemini.
        chunks: Chunks indexados.
        vectorizer: Vetorizador TF-IDF.
        matrix: Matriz TF-IDF.

    Returns:
        Resposta gerada pelo Gemini.
    """
    context, _ = search_context(
        question=question, chunks=chunks, vectorizer=vectorizer, matrix=matrix
    )

    prompt: str = build_prompt(question=question, context=context)

    response = chat.send_message(prompt)

    response_text: str | None = response.text

    if response_text is None:
        raise RuntimeError("O Gemini retornou uma resposta sem conteúdo textual.")

    return response_text


def main() -> None:
    """
    Executa o chatbot.
    """
    api_key: str | None = getenv("GEMINI_API_KEY")

    if api_key is None:
        raise RuntimeError("GEMINI_API_KEY não definida.")

    chunks: list[dict[str, str]] = load_chunks("data/chunks.json")

    documents: list[str] = [chunk["text"] for chunk in chunks]

    vectorizer, matrix = build_vectorizer(documents)

    client: Client = Client(api_key=api_key)

    chat: Chat = create_chat(client)

    while True:
        question: str = input("(Você): ").strip()

        print()

        if question == "0":
            break

        try:
            response: str = answer(
                question=question,
                chat=chat,
                chunks=chunks,
                vectorizer=vectorizer,
                matrix=matrix,
            )

            print("(IA):")
            print(response)
            print()

        except Exception as error:
            print(f"(Erro): {error}")
            print()


if __name__ == "__main__":
    main()
