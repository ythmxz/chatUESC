from json import dump, load


CHUNK_SIZE: int = 3000
OVERLAP: int = 500


def load_pages(input_path: str = "data/pages.json") -> list[dict[str, str]]:
    """Carrega páginas coletadas pelo crawler a partir do arquivo JSON."""
    with open(input_path, encoding="utf8") as f:
        return load(f)


def split_page_into_chunks(
    page: dict[str, str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[dict[str, str]]:
    """Divide o texto de uma página em blocos com sobreposição para preservar contexto."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[dict[str, str]] = []
    text: str = page["text"]
    start: int = 0

    while start < len(text):
        end: int = start + chunk_size
        chunks.append(
            {
                "url": page["url"],
                "title": page["title"],
                "text": text[start:end],
            }
        )
        start += chunk_size - overlap

    return chunks


def build_chunks(
    pages: list[dict[str, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[dict[str, str]]:
    """Gera todos os chunks do conjunto de páginas para indexação semântica."""
    chunks: list[dict[str, str]] = []
    for page in pages:
        chunks.extend(split_page_into_chunks(page, chunk_size=chunk_size, overlap=overlap))
    return chunks


def save_chunks(chunks: list[dict[str, str]], output_path: str = "chunks.json") -> None:
    """Salva os chunks em JSON para consumo do chatbot."""
    with open(output_path, "w", encoding="utf8") as f:
        dump(chunks, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Lê as páginas coletadas, gera chunks e grava o índice no disco."""
    pages = load_pages()
    chunks = build_chunks(pages)
    save_chunks(chunks)
    print(f"Generated {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
