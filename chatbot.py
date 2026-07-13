from json import load
from os import getenv
from pathlib import Path
from typing import Any, TypedDict, cast
from collections.abc import Sequence
from getpass import getpass

from dotenv import load_dotenv

from google.genai import Client
from google.genai.chats import Chat
from google.genai.types import GenerateContentConfig
from numpy import array, dtype, float64, int64, ndarray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DEBUG_RESULTS: int = 10

CHUNKS_PATH: Path = Path("data/chunks.json")
MODEL: str = getenv("GEMINI_MODEL", "gemini-3-flash-preview")
TOP_K: int = 15
MAX_CHUNKS_PER_URL: int = 5
SIMILARITY_THRESHOLD: float = 0.10

SYSTEM_INSTRUCTION: str = """
Você é o ChatUESC, um assistente sobre a Universidade Estadual de Santa Cruz.

Responda em português do Brasil e mantenha o contexto da conversa.
Não invente informações nem afirme como certo aquilo que não souber.

O contexto enviado junto à pergunta é uma fonte preferencial, mas não é uma
restrição. Quando ele estiver incompleto, irrelevante ou não contiver a resposta,
responda utilizando seu conhecimento geral ou procure normalmente. Não diga ao usuário
para procurar a informação por conta própria quando você puder fornecer uma resposta útil.
"""

def get_api_key() -> str:
    """
    Obtém a chave da API Gemini.

    Primeiro tenta carregar a variável GEMINI_API_KEY.
    Caso ela não exista, solicita a chave pelo terminal.

    Returns:
        Chave da API Gemini.

    Raises:
        RuntimeError: Caso nenhuma chave seja informada.
    """
    api_key: str | None = getenv(
        "GEMINI_API_KEY"
    )

    if api_key is not None:
        api_key = api_key.strip()

    if api_key:
        return api_key

    api_key = getpass(
        "Informe a chave da API Gemini: "
    ).strip()

    if not api_key:
        raise RuntimeError(
            "Nenhuma chave da API Gemini foi informada."
        )

    return api_key


def print_retrieval_debug(
    similarities: ndarray[Any, dtype[float64]],
    chunks: Sequence[Chunk],
    selected_chunks: Sequence[Chunk],
    selected_scores: Sequence[float],
    debug: bool,
) -> None:
    """
    Exibe informações de depuração da recuperação de contexto.

    Args:
        similarities: Similaridades entre a pergunta e os chunks.
        chunks: Chunks disponíveis no índice.
        selected_chunks: Chunks selecionados para o contexto.
        selected_scores: Scores dos chunks selecionados.
        debug: Indica se a depuração está habilitada.
    """
    if not debug:
        return

    top_indexes: ndarray[Any, dtype[int64]] = (
        similarities.argsort()[::-1][:DEBUG_RESULTS]
    )

    print()
    print("=" * 80)
    print("MELHORES RESULTADOS DO TF-IDF")
    print("=" * 80)

    for position, index_value in enumerate(
        top_indexes,
        start=1,
    ):
        index: int = int(index_value)
        chunk: Chunk = chunks[index]
        score: float = float(similarities[index])

        preview: str = (
            chunk["text"][:200]
            .replace("\n", " ")
            .strip()
        )

        print()
        print(f"{position}. Score: {score:.4f}")
        print(f"Título: {chunk['title']}")
        print(f"URL: {chunk['url']}")
        print(f"Trecho: {preview}...")

    print()
    print("=" * 80)
    print("CHUNKS ENVIADOS AO GEMINI")
    print("=" * 80)

    if not selected_chunks:
        print()
        print("Nenhum chunk foi selecionado.")
    else:
        for position, (chunk, score) in enumerate(
            zip(
                selected_chunks,
                selected_scores,
            ),
            start=1,
        ):
            preview: str = (
                chunk["text"][:200]
                .replace("\n", " ")
                .strip()
            )

            print()
            print(f"{position}. Score: {score:.4f}")
            print(f"Título: {chunk['title']}")
            print(f"URL: {chunk['url']}")
            print(f"Trecho: {preview}...")

    print()
    print("=" * 80)
    print()

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
    return [prepare_document(chunk) for chunk in chunks]


def prepare_document(chunk: Chunk) -> str:
    """
    Prepara um chunk para indexação, atribuindo maior relevância
    ao título e incluindo os termos presentes na URL.

    Args:
        chunk: Chunk que será indexado.

    Returns:
        Texto preparado para o índice TF-IDF.
    """
    normalized_url: str = (
        chunk["url"]
        .replace("https://", " ")
        .replace("http://", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace("?", " ")
        .replace("=", " ")
        .replace(".", " ")
    )

    return (
        f"{chunk['title']} "
        f"{chunk['title']} "
        f"{normalized_url} "
        f"{chunk['text']}"
    )


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
    strip_accents="unicode",
    ngram_range=(1, 2),
    sublinear_tf=True,
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


def calculate_metadata_bonus(
    question: str,
    chunk: Chunk,
) -> float:
    """
    Calcula um bônus de relevância com base no título e na URL.

    Args:
        question: Pergunta feita pelo usuário.
        chunk: Chunk avaliado.

    Returns:
        Bônus que será somado à similaridade TF-IDF.
    """
    question_terms: set[str] = set(
        question.lower().split()
    )

    metadata: str = (
        f"{chunk['title']} {chunk['url']}"
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
    )

    matching_terms: int = sum(
        term in metadata
        for term in question_terms
        if len(term) >= 4
    )

    return matching_terms * 0.03


def search_context(
    question: str,
    chunks: list[Chunk],
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    k: int = TOP_K,
    debug: bool = False,
) -> tuple[
    list[Chunk],
    ndarray[Any, dtype[float64]],
]:
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

    for index, chunk in enumerate(chunks):
        similarities[index] += calculate_metadata_bonus(
            question,
            chunk,
        )

    indexes: ndarray[Any, dtype[int64]] = similarities.argsort()[::-1]

    selected_chunks: list[Chunk] = []
    selected_scores: list[float] = []
    chunks_per_url: dict[str, int] = {}

    for index_value in indexes:
        index: int = int(index_value)
        score: float = float(similarities[index])

        if (
            score < SIMILARITY_THRESHOLD
            and selected_chunks
        ):
            break

        url: str = chunks[index]["url"]

        if chunks_per_url.get(url, 0) >= 3:
            continue

        selected_chunks.append(chunks[index])
        selected_scores.append(score)

        chunks_per_url[url] = (
            chunks_per_url.get(url, 0) + 1
        )

        if len(selected_chunks) >= k:
            break

    print_retrieval_debug(
        similarities=similarities,
        chunks=chunks,
        selected_chunks=selected_chunks,
        selected_scores=selected_scores,
        debug=debug,
    )

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
    responda utilizando seu conhecimento geral ou procure normalmente. Não responda apenas
    que a informação não foi encontrada e não mande o usuário procurá-la por conta própria.

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
    debug: bool = False,
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
        debug=debug,
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


def main(
    debug: bool = False
) -> None:
    """
    Executa o chatbot.

    Args:
        debug: Indica se os resultados da recuperação
            devem ser exibidos.
    """
    """Inicializa os componentes e executa o laço de conversa."""
    api_key: str | None = getenv(
    "GEMINI_API_KEY"
)

    if api_key is None:
        raise RuntimeError(
            "GEMINI_API_KEY não definida."
        )

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
                debug=debug,
            )

            print("(IA):")
            print(response)
            print()

        except Exception as error:
            print(f"(Erro): {error}")
            print()


if __name__ == "__main__":
    main()
