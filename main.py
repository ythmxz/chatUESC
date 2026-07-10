import argparse
from argparse import Namespace

import build_index
import chatbot
import crawler


def parse_arguments() -> Namespace:
    """
    Lê os argumentos da linha de comando.

    Returns:
        Argumentos informados pelo usuário.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Executa o pipeline do ChatUESC: crawler, geração dos chunks e chatbot."
        )
    )

    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Atualiza páginas e chunks.",
    )

    parser.add_argument(
        "-c",
        "--crawl",
        action="store_true",
        help="Não executa a coleta das páginas do site.",
    )

    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        help="Não gera novamente o arquivo de chunks.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Executa o pipeline principal do ChatUESC.
    """
    arguments: Namespace = parse_arguments()

    if arguments.update:
        print("Executando crawler...")
        print()

        crawler.main()
        print()

        print("Gerando chunks...")
        print()

        build_index.main()
        print()

    if arguments.crawl:
        print("Executando crawler...")
        print()

        crawler.main()

        print()

    if arguments.build:
        print("Gerando chunks...")
        print()

        build_index.main()

        print()

    print("Iniciando chatbot...")
    print()

    chatbot.main()


if __name__ == "__main__":
    main()
