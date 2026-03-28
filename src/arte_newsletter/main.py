import os
import textwrap
import requests
import datetime
from dateutil import relativedelta
from openai import OpenAI

today = datetime.date.today()
next_month_date = today + relativedelta.relativedelta(months=1)

next_month_name = next_month_date.strftime('%B').capitalize()
next_month_year = next_month_date.year

city_name_for_prompt = os.getenv("ART_NEWS_CITY", "Málaga")

SYSTEM_PROMPT = """
You are an expert art curator.
Your task is to generate a concise summary of art exhibitions available next month in a specific city.
Use web search to find exhibitions, shows, and art events.
Strict rules:
- The content MUST be for the upcoming calendar month. For example, if today is  any day inJune, the content must be for July.
- Focus exclusively on the city/area/state provided in the user prompt.
- Return ONLY the requested format. No introductions, no conclusions, no extra commentary.
- If there are no relevant exhibitions, return: No encontre ninguna muestra para el mes siguiente.
Output format (in Spanish):
- First line: "Muestras de arte para ver en {ciudad} en {mes}" (e.g., "Muestras de arte para ver en Málaga en Julio").
- Then, ONLY a bullet list.
- Each bullet point must contain:
- Title of the exhibition.
- A 1-2 sentence summary that includes the venue (museum, gallery, etc.).
- The URL for more information, if available.

No introduction. No conclusion. No extra commentary.
Only the title and the bullet list.
""".strip()

USER_PROMPT = f"Find and summarize the most important art exhibitions for {city_name_for_prompt} during the month of {next_month_name} {next_month_year}.".strip()


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

    # Telegram tiene límite de tamaño por mensaje; por simplicidad, partimos en bloques razonables si hiciera falta.
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