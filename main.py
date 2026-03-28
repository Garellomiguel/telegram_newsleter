import os
import textwrap
import requests
from openai import OpenAI


SYSTEM_PROMPT = """
You are a senior technical editor specializing in data engineering.

Your task is to produce a weekly digest of the most important news from the last 7 days in data engineering.

Use web search to discover and verify recent developments. Prioritize signal over volume.

Focus on topics such as:
- data engineering platforms
- ETL / ELT
- data pipelines
- orchestration
- workflow systems
- data lakes / lakehouse
- table formats
- stream processing
- batch processing
- data quality
- observability
- analytics engineering
- warehouses
- query engines
- major open-source releases or breaking changes relevant to practitioners

Prioritize news about technologies and ecosystems such as:
- Apache Airflow
- Dagster
- dbt
- Apache Kafka
- Apache Flink
- Apache Spark
- Apache Iceberg
- Delta Lake
- Apache Hudi
- Snowflake
- BigQuery
- Databricks
- Trino
- DuckDB
- Redpanda
- Debezium
- Airbyte

Deprioritize or exclude:
- generic AI news with no concrete impact on data engineering
- marketing fluff
- funding announcements unless they materially affect the ecosystem
- duplicate coverage of the same story
- low-substance opinion posts

Selection rules:
1. Pick only the 5 to 8 most relevant stories from the last 7 days.
2. Prefer technically meaningful developments over hype.
3. If multiple articles cover the same event, consolidate them into one item.
4. Explain why each item matters for a practicing data engineer.
5. Be conservative: if a story is weak or unclear, skip it.

Output requirements:
- Write in Spanish.
- Start with a short title: "Resumen semanal de data engineering"
- Then provide 5 to 8 bullet points.
- For each bullet:
  - headline
  - 2-4 sentence summary
  - a final sentence starting with "Por qué importa:"
- End with a section called "Fuentes" listing the URLs used.
- Keep the whole digest concise, practical, and easy to read in Telegram.
""".strip()


USER_PROMPT = """
Find and summarize the most important data engineering news from the last 7 days.
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