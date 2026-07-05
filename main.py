from argparse import ArgumentParser, Namespace
from os import getenv

from build_index import build_chunks, load_pages, save_chunks
from chatbot import API_KEY, run_chat
from crawler import crawl_pages, save_pages


def run_pipeline(skip_crawl: bool = False, skip_index: bool = False) -> None:
    """Executa pipeline completo (coleta, indexação e chat) com etapas opcionais."""
    if skip_crawl:
        pages = load_pages()
        print(f"Loaded {len(pages)} pages from data/pages.json.")
    else:
        print("Starting crawler...")
        pages = crawl_pages()
        save_pages(pages)
        print(f"Collected {len(pages)} pages.")

    if skip_index:
        print("Skipping chunk generation.")
    else:
        print("Building chunks...")
        chunks = build_chunks(pages)
        save_chunks(chunks)
        print(f"Generated {len(chunks)} chunks.")

    api_key = getenv("GOOGLE_API_KEY") or API_KEY
    if not api_key:
        raise ValueError(
            "Configure GOOGLE_API_KEY no ambiente ou preencha API_KEY em chatbot.py"
        )

    print("Starting chatbot. Enter 0 to exit.")
    run_chat(api_key=api_key)


def parse_args() -> Namespace:
    """Define e processa argumentos de linha de comando do script principal."""
    parser = ArgumentParser(description="Pipeline do ChatUESC")
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Pula coleta e usa data/pages.json existente.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Pula geração de chunks e usa chunks.json existente.",
    )

    return parser.parse_args()


def main() -> None:
    """Integra o fluxo de execução do projeto em um único comando."""
    args = parse_args()
    run_pipeline(skip_crawl=args.skip_crawl, skip_index=args.skip_index)


if __name__ == "__main__":
    main()
