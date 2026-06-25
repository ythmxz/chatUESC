from collections import deque
from bs4 import BeautifulSoup
from json import dump
from requests import get
from time import sleep
from urllib.parse import ParseResult, urljoin, urlparse


START_URL: str = "https://www.uesc.br/"
MAX_PAGES: int = 150


def is_valid_url(url: str) -> bool:
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
    keywords: tuple = (
        "arint",
        "noticias",
        "eventos",
        "player",
        "portalsagres"
    )

    url = url.lower()

    return any(word in url for word in keywords)


priority_queue: deque[str] = deque([START_URL])
secondary_queue: deque[str] = deque()

visited: set[str] = set()
queued: set[str] = {START_URL}
pages: list[dict[str, str]] = []

while priority_queue or secondary_queue and len(visited) < MAX_PAGES:
    if priority_queue:
        url = priority_queue.popleft()
    else:
        url = secondary_queue.popleft()

    if url in visited:
        continue

    print(f"[{len(visited)+1}/{MAX_PAGES}] {url}")

    visited.add(url)

    try:
        response = get(
            url,
            timeout=10,
            headers={
                "User-Agent":
                "ChatUESCBot/1.0"
            }
        )

        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script",
            "style",
            "noscript"
        ]):
            tag.decompose()

        title = (soup.title.text.strip() if soup.title else "Untitled")

        text = soup.get_text(separator=" ", strip=True)

        pages.append({
            "url": url,
            "title": title,
            "text": text
        })

        for link in soup.find_all("a", href=True):
            new_url = urljoin(url, str(link["href"]))
            new_url = new_url.split("#")[0]

            if (is_valid_url(new_url) and new_url not in visited and new_url not in queued):
                if is_secondary_url(new_url):
                    secondary_queue.append(new_url)
                else:
                    priority_queue.append(new_url)

                queued.add(new_url)

    except Exception as e:
        print(f"Error: {e}")

    sleep(0.5)

with open("data/pages.json", "w", encoding="utf8") as f:
    dump(pages, f, ensure_ascii=False, indent=2)

print(f"\nCollected {len(pages)} pages.")

# from bs4 import BeautifulSoup
# from json import dump
# from requests import get

# URLS: list[str] = [
#     "https://www.uesc.br/",
#     "https://www.uesc.br/cursos/graduacao/",
#     "https://www.uesc.br/prograd/",
# ]

# data: list[dict[str, str]] = []

# for url in URLS:
#     try:
#         html: str = get(url, timeout=10).text
#         soup: BeautifulSoup = BeautifulSoup(html, "html.parser")

#         title: str = soup.title.text.strip() if soup.title else "Untitled"
#         text: str = soup.get_text(separator=" ", strip=True)

#         data.append({
#             "url": url,
#             "title": title,
#             "text": text
#         })

#         print(f"Collected: {url}")

#     except Exception as e:
#         print(f"Error: {e}")

# with open("data/pages.json", "w", encoding="utf8") as f:
#     dump(data, f, ensure_ascii=False, indent=2)
