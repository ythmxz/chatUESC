from collections import deque
from json import dump
from pathlib import Path
from time import sleep
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)

import requests
from bs4 import BeautifulSoup
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

START_URL: str = "https://www.uesc.br/"

MAX_PAGES: int = 100
REQUEST_DELAY: float = 0.5
REQUEST_TIMEOUT: int = 10

OUTPUT_PATH: Path = Path("data/pages.json")

BLOCKED_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
)

TRACKING_PARAMETERS: set[str] = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}

HIGH_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "graduacao",
    "prograd",
    "colegiado",
    "departamento",
    "pesquisa",
    "propp",
    "extensao",
    "proex",
    "biblioteca",
    "curso",
    "mestrado",
    "doutorado",
    "calendario",
    "laboratorio",
)

LOW_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "noticia",
    "noticias",
    "evento",
    "eventos",
    "agenda",
    "mural",
    "edital",
    "editais",
    "portaria",
)


def create_session() -> Session:
    """
    Cria uma sessão HTTP reutilizável.

    Returns:
        Sessão configurada com retry automático.
    """
    session: Session = requests.Session()

    retry: Retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET",
        ],
    )

    adapter: HTTPAdapter = HTTPAdapter(max_retries=retry)

    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": ("ChatUESC/1.0 (Academic Research Bot)")})

    return session


def normalize_url(url: str) -> str:
    """
    Normaliza URLs removendo fragmentos,
    parâmetros de rastreamento e barras finais.

    Args:
        url: URL original.

    Returns:
        URL normalizada.
    """
    parsed = urlparse(url)

    filtered_query: list[tuple[str, str]] = [
        (key, value)
        for key, value in parse_qsl(parsed.query)
        if key.lower() not in TRACKING_PARAMETERS
    ]

    path: str = parsed.path.rstrip("/")

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc.lower(),
            path,
            "",
            urlencode(filtered_query),
            "",
        )
    )


def is_valid_url(url: str) -> bool:
    """
    Verifica se uma URL pode ser visitada.

    Args:
        url: URL analisada.

    Returns:
        True se a URL for válida.
    """
    parsed = urlparse(url)

    if parsed.scheme not in (
        "http",
        "https",
    ):
        return False

    if "uesc.br" not in parsed.netloc:
        return False

    return not parsed.path.lower().endswith(BLOCKED_EXTENSIONS)


def get_priority(url: str) -> int:
    """
    Define a prioridade de exploração.

    0 = alta prioridade
    1 = prioridade normal
    2 = baixa prioridade

    Args:
        url: URL analisada.

    Returns:
        Nível de prioridade.
    """
    lowered_url: str = url.lower()

    if any(keyword in lowered_url for keyword in LOW_PRIORITY_KEYWORDS):
        return 2

    if any(keyword in lowered_url for keyword in HIGH_PRIORITY_KEYWORDS):
        return 0

    return 1


def extract_text(soup: BeautifulSoup) -> str:
    """
    Extrai o texto principal da página.

    Args:
        soup: Documento HTML.

    Returns:
        Texto extraído.
    """
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "header",
            "footer",
        ]
    ):
        tag.decompose()

    content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.body
    )

    if content is None:
        return ""

    return content.get_text(
        separator=" ",
        strip=True,
    )


def extract_title(
    soup: BeautifulSoup,
    url: str,
) -> str:
    """
    Extrai o título da página.

    Args:
        soup: Documento HTML.
        url: URL da página.

    Returns:
        Título encontrado.
    """
    if soup.title is not None and soup.title.string is not None:
        return soup.title.string.strip()

    return url


def extract_links(
    soup: BeautifulSoup,
    base_url: str,
) -> list[str]:
    """
    Extrai todos os links válidos
    encontrados na página.

    Args:
        soup: Documento HTML.
        base_url: URL da página atual.

    Returns:
        Lista de URLs encontradas.
    """
    links: list[str] = []

    for tag in soup.find_all(
        "a",
        href=True,
    ):
        href = tag.get("href")

        if not isinstance(href, str):
            continue

        url: str = normalize_url(
            urljoin(
                base_url,
                href,
            )
        )

        if is_valid_url(url):
            links.append(url)

    return links


def fetch_page(
    session: Session,
    url: str,
) -> BeautifulSoup | None:
    """
    Realiza o download de uma página.

    Args:
        session: Sessão HTTP.
        url: URL da página.

    Returns:
        Documento HTML ou None.
    """
    try:
        response: Response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            return None

        return BeautifulSoup(
            response.text,
            "html.parser",
        )

    except requests.RequestException:
        return None


def save_pages(pages: list[dict[str, str]]) -> None:
    """
    Salva as páginas coletadas.

    Args:
        pages: Conteúdo coletado.
    """
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf8",
    ) as file:
        dump(
            pages,
            file,
            ensure_ascii=False,
            indent=2,
        )


def crawl() -> list[dict[str, str]]:
    """
    Executa o crawler BFS priorizado.

    Returns:
        Lista de páginas coletadas.
    """
    session: Session = create_session()

    queues: list[deque[str]] = [
        deque([START_URL]),
        deque(),
        deque(),
    ]

    visited: set[str] = set()
    queued: set[str] = {START_URL}
    pages: list[dict[str, str]] = []

    while len(visited) < MAX_PAGES:
        current_url: str | None = None

        for queue in queues:
            if queue:
                current_url = queue.popleft()
                break

        if current_url is None:
            break

        if current_url in visited:
            continue

        visited.add(current_url)

        print(f"[{len(visited)}/{MAX_PAGES}] {current_url}")

        soup = fetch_page(
            session,
            current_url,
        )

        if soup is None:
            continue

        text: str = extract_text(soup)

        if len(text) >= 100:
            pages.append(
                {
                    "url": current_url,
                    "title": extract_title(
                        soup,
                        current_url,
                    ),
                    "text": text,
                }
            )

        for link in extract_links(
            soup,
            current_url,
        ):
            if link in visited:
                continue

            if link in queued:
                continue

            queued.add(link)

            priority: int = get_priority(link)

            queues[priority].append(link)

        sleep(REQUEST_DELAY)

    return pages


def main() -> None:
    """
    Executa o crawler.
    """
    pages: list[dict[str, str]] = crawl()

    save_pages(pages)

    print()
    print(f"{len(pages)} páginas coletadas.")
    print(f"Arquivo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
