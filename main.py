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
            "Executa o ChatUESC e, opcionalmente, "
            "atualiza as páginas e os chunks."
        )
    )

    parser.add_argument(
        "-c",
        "--crawl",
        action="store_true",
        help="Coleta novamente as páginas do site da UESC.",
    )

    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        help="Gera novamente os chunks a partir das páginas coletadas.",
    )

    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Executa o crawler e gera novamente os chunks.",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Exibe os resultados da recuperação TF-IDF.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Executa as etapas solicitadas e inicia o chatbot.
    """
    arguments: Namespace = parse_arguments()

    should_crawl: bool = (
        arguments.crawl
        or arguments.update
    )

    should_build: bool = (
        arguments.build
        or arguments.update
    )

    if should_crawl:
        print("Executando crawler...")
        print()

        crawler.main()

        print()

    if should_build:
        print("Gerando chunks...")
        print()

        build_index.main()

        print()

    print("Iniciando chatbot...")
    print()

    chatbot.main(
        debug=arguments.debug
    )


if __name__ == "__main__":
    main()
