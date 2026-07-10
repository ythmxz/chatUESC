from json import dump, load
from pathlib import Path
from typing import TypedDict

INPUT_PATH: Path = Path("data/pages.json")
OUTPUT_PATH: Path = Path("data/chunks.json")

CHUNK_SIZE: int = 3000
OVERLAP: int = 500
MIN_CHUNK_SIZE: int = 100


class Page(TypedDict):
    """
    Representa uma página coletada pelo crawler.
    """

    url: str
    title: str
    text: str


class Chunk(TypedDict):
    """
    Representa um trecho de texto indexado.
    """

    url: str
    title: str
    text: str


def load_pages(path: Path) -> list[Page]:
    """
    Carrega as páginas coletadas pelo crawler.

    Args:
        path: Caminho do arquivo JSON de entrada.

    Returns:
        Lista de páginas carregadas.

    Raises:
        FileNotFoundError: Caso o arquivo não exista.
        ValueError: Caso o arquivo não contenha uma lista.
    """
    with open(
        path,
        "r",
        encoding="utf8",
    ) as file:
        data: object = load(file)

    if not isinstance(data, list):
        raise ValueError("O arquivo de páginas deve conter uma lista.")

    pages: list[Page] = []

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

        pages.append(
            {
                "url": url,
                "title": title,
                "text": text,
            }
        )

    return pages


def normalize_text(text: str) -> str:
    """
    Normaliza espaços em branco no texto.

    Args:
        text: Texto original.

    Returns:
        Texto com espaços normalizados.
    """
    return " ".join(text.split())


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """
    Divide um texto em chunks com sobreposição.

    Args:
        text: Texto a ser dividido.
        chunk_size: Tamanho máximo de cada chunk.
        overlap: Quantidade de caracteres repetidos
            entre chunks consecutivos.

    Returns:
        Lista de chunks gerados.

    Raises:
        ValueError: Caso os parâmetros sejam inválidos.
    """
    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE deve ser maior que zero.")

    if overlap < 0:
        raise ValueError("OVERLAP não pode ser negativo.")

    if overlap >= chunk_size:
        raise ValueError("OVERLAP deve ser menor que CHUNK_SIZE.")

    normalized_text: str = normalize_text(text)

    if len(normalized_text) < MIN_CHUNK_SIZE:
        return []

    chunks: list[str] = []
    start: int = 0

    while start < len(normalized_text):
        end: int = min(
            start + chunk_size,
            len(normalized_text),
        )

        chunk: str = normalized_text[start:end].strip()

        if end < len(normalized_text) and chunk:
            last_space: int = chunk.rfind(" ")

            if last_space > chunk_size // 2:
                end = start + last_space
                chunk = normalized_text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)

        if end >= len(normalized_text):
            break

        start = end - overlap

    return chunks


def build_chunks(
    pages: list[Page],
) -> list[Chunk]:
    """
    Gera chunks para todas as páginas válidas.

    Args:
        pages: Páginas coletadas pelo crawler.

    Returns:
        Lista de chunks com URL, título e texto.
    """
    chunks: list[Chunk] = []

    for page in pages:
        page_chunks: list[str] = split_text(page["text"])

        for text in page_chunks:
            chunks.append(
                {
                    "url": page["url"],
                    "title": page["title"],
                    "text": text,
                }
            )

    return chunks


def save_chunks(
    chunks: list[Chunk],
    path: Path,
) -> None:
    """
    Salva os chunks em um arquivo JSON.

    Args:
        chunks: Lista de chunks gerados.
        path: Caminho do arquivo de saída.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf8",
    ) as file:
        dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    """
    Executa a geração do índice textual.
    """
    pages: list[Page] = load_pages(INPUT_PATH)

    chunks: list[Chunk] = build_chunks(pages)

    save_chunks(
        chunks,
        OUTPUT_PATH,
    )

    print(f"{len(pages)} páginas processadas.")
    print(f"{len(chunks)} chunks gerados.")
    print(f"Arquivo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
