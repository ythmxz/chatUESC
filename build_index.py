from json import dump, load


CHUNK_SIZE: int = 3000
OVERLAP: int = 500


with open("data/pages.json", encoding="utf8") as f:
    pages: list[dict[str, str]] = load(f)

chunks: list[dict[str, str]] = []

for page in pages:
    text: str = page["text"]
    start: int = 0

    while start < len(text):
        end: int = start + CHUNK_SIZE

        chunks.append({
            "url": page["url"],
            "title": page["title"],
            "text": text[start:end]
        })

        start += CHUNK_SIZE - OVERLAP

with open("chunks.json", "w", encoding="utf8") as f:
    dump(chunks, f, ensure_ascii=False, indent=2)
