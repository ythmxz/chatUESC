from collections import deque
from bs4 import BeautifulSoup
from json import dump
from os import makedirs
from os.path import dirname
from requests import get
from time import sleep
from urllib.parse import ParseResult, urljoin, urlparse


START_URL: str = "https://www.uesc.br/"
MAX_PAGES: int = 150
REQUEST_TIMEOUT: int = 10
CRAWL_DELAY_SECONDS: float = 0.5


def is_valid_url(url: str) -> bool:
    """Valida se uma URL pertence ao domínio alvo e não aponta para arquivo binário."""
    parsed: ParseResult = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False

    if "uesc.br" not in parsed.netloc:
        return False

    blocked: tuple = (
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".zip",
        ".rar"
    )

    return not url.lower().endswith(blocked)


def is_secondary_url(url: str) -> bool:
    """Classifica URLs de baixa prioridade para visitação após links principais."""
    keywords: tuple = (
        "arint",
        "noticias",
        "eventos",
        "player",
        "portalsagres"
    )

    url = url.lower()

    return any(word in url for word in keywords)


def crawl_pages(
    start_url: str = START_URL,
    max_pages: int = MAX_PAGES,
    delay_seconds: float = CRAWL_DELAY_SECONDS,
) -> list[dict[str, str]]:
    """Rastreia páginas da UESC e retorna metadados com URL, título e texto limpo."""
    priority_queue: deque[str] = deque([start_url])
    secondary_queue: deque[str] = deque()

    visited: set[str] = set()
    queued: set[str] = {start_url}
    pages: list[dict[str, str]] = []

    while (priority_queue or secondary_queue) and len(visited) < max_pages:
        url = priority_queue.popleft() if priority_queue else secondary_queue.popleft()

        if url in visited:
            continue

        print(f"[{len(visited) + 1}/{max_pages}] {url}")
        visited.add(url)

        try:
            response = get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "ChatUESCBot/1.0"},
            )

            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            title = soup.title.text.strip() if soup.title else "Untitled"
            text = soup.get_text(separator=" ", strip=True)

            pages.append({"url": url, "title": title, "text": text})

            for link in soup.find_all("a", href=True):
                new_url = urljoin(url, str(link["href"])).split("#")[0]

                if is_valid_url(new_url) and new_url not in visited and new_url not in queued:
                    if is_secondary_url(new_url):
                        secondary_queue.append(new_url)
                    else:
                        priority_queue.append(new_url)

                    queued.add(new_url)

        except Exception as exc:
            print(f"Error: {exc}")

        sleep(delay_seconds)

    return pages


def save_pages(pages: list[dict[str, str]], output_path: str = "data/pages.json") -> None:
    """Salva em disco as páginas coletadas no formato JSON."""
    output_dir = dirname(output_path)
    if output_dir:
        makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        dump(pages, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Executa o crawler com configuração padrão e salva o resultado em disco."""
    pages = crawl_pages()
    save_pages(pages)
    print(f"\nCollected {len(pages)} pages.")


if __name__ == "__main__":
    main()
