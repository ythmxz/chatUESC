from json import load
from os import getenv
from pathlib import Path
from typing import Any, TypedDict, cast

from dotenv import load_dotenv

from google.genai import Client
from google.genai.chats import Chat
from google.genai.types import GenerateContentConfig
from numpy import array, dtype, float64, int64, ndarray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

CHUNKS_PATH: Path = Path("data/chunks.json")
MODEL: str = getenv("GEMINI_MODEL", "gemini-3-flash-preview")
TOP_K: int = 10
MAX_CHUNKS_PER_URL: int = 3
SIMILARITY_THRESHOLD: float = 0.10

SYSTEM_INSTRUCTION: str = """
Você é o ChatUESC, um assistente sobre a Universidade Estadual de Santa Cruz.

Responda em português do Brasil e mantenha o contexto da conversa.
Não invente informações nem afirme como certo aquilo que não souber.

O contexto enviado junto à pergunta é uma fonte preferencial, mas não é uma
restrição. Quando ele estiver incompleto, irrelevante ou não contiver a resposta,
responda utilizando seu conhecimento geral. Não diga ao usuário para procurar a
informação por conta própria quando você puder fornecer uma resposta útil.
"""


class Chunk(TypedDict):
    """Representa um trecho de uma página coletada."""

    url: str
    title: str
    text: str


def load_chunks(path: Path) -> list[Chunk]:
    """
    Carrega e valida os chunks armazenados em um arquivo JSON.

    Args:
        path: Caminho do arquivo contendo os chunks.

    Returns:
        Lista de chunks válidos.

    Raises:
        ValueError: Caso o conteúdo do arquivo não seja uma lista ou não
            contenha nenhum chunk válido.
    """
    with open(path, "r", encoding="utf8") as file:
        data: object = load(file)

    if not isinstance(data, list):
        raise ValueError("O arquivo de chunks deve conter uma lista.")

    chunks: list[Chunk] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        url: object = item.get("url")
        title: object = item.get("title")
        text: object = item.get("text")

        if not isinstance(url, str):
            continue

        if not isinstance(title, str):
            continue

        if not isinstance(text, str):
            continue

        chunks.append({"url": url, "title": title, "text": text})

    if not chunks:
        raise ValueError("Nenhum chunk válido foi encontrado.")

    return chunks


def build_documents(chunks: list[Chunk]) -> list[str]:
    """
    Monta os documentos usados no índice TF-IDF.

    O título é incluído no documento para que páginas com títulos relevantes,
    como "Cursos de Graduação", sejam recuperadas com maior facilidade.

    Args:
        chunks: Chunks que serão indexados.

    Returns:
        Lista de textos utilizados pelo vetorizador.
    """
    return [f"{chunk['title']} {chunk['text']}" for chunk in chunks]


def build_vectorizer(
    documents: list[str],
) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Constrói o índice TF-IDF dos documentos.

    Args:
        documents: Textos utilizados na construção do índice.

    Returns:
        Vetorizador treinado e matriz TF-IDF dos documentos.

    Raises:
        ValueError: Caso a lista de documentos esteja vazia.
    """
    if not documents:
        raise ValueError("Não é possível construir o índice sem documentos.")

    vectorizer: TfidfVectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )

    matrix: csr_matrix = cast(
        csr_matrix,
        vectorizer.fit_transform(documents),
    )

    return vectorizer, matrix


def create_chat(client: Client) -> Chat:
    """
    Cria uma sessão de conversa persistente com o Gemini.

    Args:
        client: Cliente da API Gemini.

    Returns:
        Sessão de chat que mantém o histórico da conversa.
    """
    return client.chats.create(
        model=MODEL,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )


def search_context(
    question: str,
    chunks: list[Chunk],
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    k: int = TOP_K,
) -> tuple[list[Chunk], ndarray[Any, dtype[float64]]]:
    """
    Recupera os chunks mais relevantes para uma pergunta.

    São permitidos vários chunks da mesma página para que listas ou conteúdos
    longos, divididos em diferentes trechos, possam ser recuperados por inteiro.

    Args:
        question: Pergunta do usuário.
        chunks: Chunks disponíveis para recuperação.
        vectorizer: Vetorizador TF-IDF treinado.
        matrix: Matriz TF-IDF dos chunks.
        k: Quantidade máxima de chunks retornados.

    Returns:
        Chunks selecionados e seus respectivos valores de similaridade.
    """
    question_vector: csr_matrix = cast(
        csr_matrix,
        vectorizer.transform([question]),
    )

    similarities: ndarray[Any, dtype[float64]] = cosine_similarity(
        question_vector,
        matrix,
    )[0]

    indexes: ndarray[Any, dtype[int64]] = similarities.argsort()[::-1]

    selected_chunks: list[Chunk] = []
    selected_scores: list[float] = []
    chunks_per_url: dict[str, int] = {}

    for index in indexes:
        score: float = float(similarities[index])

        if score < SIMILARITY_THRESHOLD:
            break

        chunk: Chunk = chunks[int(index)]
        url: str = chunk["url"]
        url_count: int = chunks_per_url.get(url, 0)

        if url_count >= MAX_CHUNKS_PER_URL:
            continue

        selected_chunks.append(chunk)
        selected_scores.append(score)
        chunks_per_url[url] = url_count + 1

        if len(selected_chunks) >= k:
            break

    return selected_chunks, array(selected_scores, dtype=float64)


def build_prompt(question: str, context: list[Chunk]) -> str:
    """
    Constrói a mensagem enviada ao Gemini.

    O contexto é apresentado como informação preferencial. O modelo também é
    instruído a responder normalmente quando os trechos não forem suficientes.

    Args:
        question: Pergunta do usuário.
        context: Chunks recuperados pelo TF-IDF.

    Returns:
        Mensagem formatada para a sessão de chat.
    """
    if not context:
        return question

    context_text: str = "\n\n".join(
        (f"Título: {chunk['title']}\nURL: {chunk['url']}\nConteúdo: {chunk['text']}")
        for chunk in context
    )

    return f"""
    Use as informações abaixo quando forem relevantes para responder à pergunta.
    Elas podem estar incompletas ou conter trechos que não respondem diretamente ao
    que foi perguntado. Nesse caso, ignore os trechos irrelevantes e complemente ou
    responda utilizando seu conhecimento geral. Não responda apenas que a informação
    não foi encontrada e não mande o usuário procurá-la por conta própria.

    CONTEXTO RECUPERADO:
    {context_text}

    PERGUNTA:
    {question}
    """.strip()


def answer(
    question: str,
    chat: Chat,
    chunks: list[Chunk],
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
) -> str:
    """
    Recupera contexto e gera uma resposta para o usuário.

    Args:
        question: Pergunta do usuário.
        chat: Sessão de conversa com o Gemini.
        chunks: Chunks disponíveis para recuperação.
        vectorizer: Vetorizador TF-IDF treinado.
        matrix: Matriz TF-IDF dos chunks.

    Returns:
        Texto produzido pelo Gemini.

    Raises:
        RuntimeError: Caso a resposta não contenha texto.
    """
    context, _ = search_context(
        question=question,
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
    )

    prompt: str = build_prompt(
        question=question,
        context=context,
    )

    response = chat.send_message(prompt)
    response_text: str | None = response.text

    if response_text is None:
        raise RuntimeError("O Gemini retornou uma resposta sem conteúdo textual.")

    return response_text


def main() -> None:
    """Inicializa os componentes e executa o laço de conversa."""
    api_key: str | None = getenv("GEMINI_API_KEY")

    if api_key is None:
        raise RuntimeError("GEMINI_API_KEY não definida.")

    chunks: list[Chunk] = load_chunks(CHUNKS_PATH)
    documents: list[str] = build_documents(chunks)
    vectorizer, matrix = build_vectorizer(documents)

    client: Client = Client(api_key=api_key)
    chat: Chat = create_chat(client)

    while True:
        question: str = input("(Você): ").strip()
        print()

        if question == "0":
            break

        if not question:
            continue

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
