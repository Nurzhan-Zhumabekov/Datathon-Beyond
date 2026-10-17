"""Optional LLM layer: turns the aggregated report into a pitch draft.

Privacy: only report.md goes out (aggregates and rule texts, never raw rows). The same prompt is always
saved to out/story_prompt.md, so it can be pasted into any chat if API use is not allowed or no key is set.
"""

from __future__ import annotations

import os

SYSTEM = (
    "Ты помогаешь команде дататона подготовить питч. Используй ТОЛЬКО цифры и факты из отчёта. "
    "Ничего не выдумывай: если для вывода не хватает данных, пиши «нужно проверить». "
    "Пиши по-русски, коротко, без громких слов про ИИ."
)

TASK = """Ниже автоматический отчёт анализа данных. Подготовь:
1. Крючок: одна цифра и одна фраза для первого слайда.
2. Главная находка: 2 предложения. Выбери ОДНУ, самую сильную и подтверждённую (stable = да).
3. Семь слайдов: заголовок-вывод и одна строка содержания для каждого.
4. Сценарий выступления на 5 минут.
5. Пять вероятных вопросов жюри с короткими ответами на основе отчёта.
6. Что команде проверить руками, прежде чем это утверждать (допущения, подозрительные колонки, малые сегменты).
"""


def build_prompt(report_md: str, context: str | None = None) -> str:
    body = "\n".join(l for l in report_md.splitlines() if not l.lstrip().startswith("!["))
    ctx = f"Контекст задачи от команды: {context}\n\n" if context else ""
    return f"{SYSTEM}\n\n{TASK}\n{ctx}--- ОТЧЁТ ---\n{body}\n"


def generate(prompt: str, model: str | None = None) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("не задана переменная окружения ANTHROPIC_API_KEY")
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model or os.environ.get("SH_LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
