from google.genai import Client
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from json import load
from numpy import dtype, float64, int64, ndarray
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Iterator


API_KEY: str = ""


def search_context(question: str, k: int = 15) -> list[dict[str, str]]:
    question_vector: spmatrix = vectorizer.transform([question])
    similarities: ndarray[Any, dtype[float64]] = cosine_similarity(question_vector, X)[0]
    indexes: ndarray[Any, dtype[int64]] = similarities.argsort()[-k:][::-1]

    results: list[dict[str, str]] = []
    used_urls: set[str] = set()

    for index in indexes:
        url: str = chunks[index]["url"]

        if url in used_urls:
            continue

        used_urls.add(url)
        results.append(chunks[index])

        if len(results) >= k:
            break

    return results


def answer(question) -> Iterator[GenerateContentResponse]:
    context: list[dict[str, str]] = search_context(question)
    context_text = "\n\n".join(item["text"] for item in context)

    prompt: str = f"""
    CONTEXTO:
    {context_text}

    INSTRUÇÕES:

    - Use somente o contexto.
    - Se houver uma lista completa no contexto, reproduza a lista completa.
    - Não resuma listas.
    - Não omita itens.

    PERGUNTA:
    {question}
    """

    response = client.models.generate_content_stream(
        model="gemini-3.5-flash",
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction="""
            Você é um assistente da UESC.

            Priorize o contexto recebido.

            Se a resposta não estiver no contexto,
            informe que não encontrou a informação.

            Não informe que está buscando ou que a resposta é baseada em um contexto recebido.
            """
        ))

    return response


with open("chunks.json", encoding="utf8") as f:
    chunks: list[dict[str, str]] = load(f)

documents: list[str] = [chunk["text"] for chunk in chunks]

vectorizer: TfidfVectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
X: spmatrix = vectorizer.fit_transform(documents)

client = Client(api_key=API_KEY)

while True:
    question = input("(Você): ").strip()
    print()

    if question == "0":
        break

    response = answer(question)

    print("(IA): ", end=" ")
    for chunk in response:
        print(chunk.text, end=" ")

    print(end="\n\n")
