import os
import textwrap
import requests
from openai import OpenAI


SYSTEM_PROMPT = """
You are a data engineering news curator.
Your task is to produce a concise bi-weekly digest of the most relevant data engineering news from the last 15 days.
Use web search to find recent information. Prioritize signal over volume.
Include only truly relevant news about data engineering (pipelines, ETL/ELT, orchestration, streaming, lakehouse, data quality, analytics engineering).
Strict rules:
- Return only 3 to 6 items
- If there are not enough relevant stories, return fewer items
- If there are no relevant stories, return an empty list
- Do NOT include content older than 7 days
- Do NOT include opinions, explanations, or meta commentary
- Do NOT include phrases like "low signal", "this week", or suggestions
- Do NOT add any text outside the required format
Output format (in Spanish):
- First line: "Resumen semanal de data engineering"
- Then ONLY a bullet list
- Each bullet must contain:
  - Short title
  - 1–2 sentence summary
  - URL on the same line or next line

No introduction. No conclusion. No extra commentary.
Only the title and the bullet list.
""".strip()

USER_PROMPT = """
Find and summarize the most important data engineering news from the last 15 days.
Make sure the stories are genuinely recent and relevant.
""".strip()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_openai_client() -> OpenAI:
    api_key = require_env("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def generate_digest() -> str:
    client = build_openai_client()

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
        instructions=SYSTEM_PROMPT,
        input=USER_PROMPT,
        tools=[{"type": "web_search"}],
    )

    text = response.output_text.strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")

    return text


def send_telegram_message(text: str) -> None:
    token = require_env("TELEGRAM_BOT_TOKEN")
    chat_id = require_env("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Telegram tiene límite de tamaño por mensaje; por simplicidad,
    # partimos en bloques razonables si hiciera falta.
    chunks = split_text(text, max_len=3500)

    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()


def split_text(text: str, max_len: int = 3500) -> list[str]:
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    current = []

    for paragraph in text.split("\n"):
        candidate = "\n".join(current + [paragraph]).strip()
        if candidate and len(candidate) > max_len:
            if current:
                parts.append("\n".join(current).strip())
                current = [paragraph]
            else:
                wrapped = textwrap.wrap(paragraph, width=max_len)
                parts.extend(wrapped[:-1])
                current = [wrapped[-1]]
        else:
            current.append(paragraph)

    if current:
        parts.append("\n".join(current).strip())

    return [p for p in parts if p]


def main() -> None:
    digest = generate_digest()
    send_telegram_message(digest)
    print("Weekly digest sent successfully.")


if __name__ == "__main__":
    main()