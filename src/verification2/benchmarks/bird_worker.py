"""BIRD SQLite episodes. The label is result-set execution accuracy."""

import sqlite3
import sys
import time
from pathlib import Path

from qera.adapters.text_env import convert_messages
from verification2.benchmarks.text_loop import run, tool
from verification2.data.split import ROOT, load_task

SYSTEM = (
    "Answer the question with SQLite. The schema is below. "
    "Call query with one read-only SELECT to inspect rows. "
    "Call submit once with the final SELECT. Do not modify the database."
)
LIMIT = 8
QUERY = tool("query", "One read-only SELECT. Returns at most 30 rows.")
SUBMIT = tool("submit", "The final SELECT. Ends the episode.")


def _database(db_id, partition):
    folder = "dev_databases" if partition == "test" else "train_databases"
    matches = sorted(
        path
        for path in (ROOT / "raw/bird/databases").rglob(db_id + ".sqlite")
        if folder in path.parts
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{db_id}: found {len(matches)} sqlite files in {folder}"
        )
    return matches[0]


def _statement(sql):
    text = (sql or "").strip()
    if text.endswith(";"):
        text = text[:-1].strip()
    if not text or ";" in text:
        raise ValueError("Send one SQL statement")
    head = text.lstrip("(").split(None, 1)[0].lower()
    if head not in ("select", "with"):
        raise ValueError("Only a read-only SELECT is allowed")
    return text


def _execute(con, sql, limit=None):
    text = _statement(sql)
    deadline = time.monotonic() + 30

    def handler():
        return 1 if time.monotonic() > deadline else 0

    con.set_progress_handler(handler, 10000)
    try:
        cursor = con.execute(text)
        rows = cursor.fetchmany(limit) if limit else cursor.fetchall()
    finally:
        con.set_progress_handler(None, 0)
    return rows


def _format(rows):
    if not rows:
        return "(no rows)"
    lines = [" | ".join(str(cell) for cell in row) for row in rows]
    return "\n".join(lines)


def _schema(con):
    rows = con.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND type IN ('table', 'view') "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return "\n\n".join(sql for (sql,) in rows)


def collect(job):
    row = load_task(job.task)
    path = _database(row["db_id"], row.get("partition"))
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        question = row["question"]
        evidence = (row.get("evidence") or "").strip()
        prompt = question if not evidence else question + "\n\nEvidence: " + evidence
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt + "\n\nSchema:\n" + _schema(con)},
        ]
        state = {"correct": False}

        def handle(name, command):
            if name == "submit":
                try:
                    predicted = _execute(con, command)
                except (sqlite3.Error, ValueError):
                    state["correct"] = False
                    return "Query failed and was not accepted.", True
                gold = _execute(con, row["SQL"])
                # Official execution accuracy: row order and duplicate rows do not matter.
                state["correct"] = set(predicted) == set(gold)
                return "Query submitted.", True
            try:
                rows = _execute(con, command, limit=31)
            except (sqlite3.Error, ValueError) as exc:
                return type(exc).__name__ + ": " + str(exc).split("\n", 1)[0][
                    :300
                ], False
            extra = "\n(more rows omitted)" if len(rows) > 30 else ""
            return _format(rows[:30]) + extra, False

        run(job, messages, [QUERY, SUBMIT], handle, LIMIT)
        correct = state["correct"]
    finally:
        con.close()
    trajectory = convert_messages(
        messages,
        benchmark="bird",
        task_domain=job.task.domain,
        task_id=job.task.task_id,
        model=job.model,
        run_id=job.output.name,
        correct=correct,
        score=1.0 if correct else 0.0,
        detail={"db_id": row["db_id"], "execution_match": correct},
        agent="bird_sqlite",
        label_source="unit_tests",
    )
    (job.output / "trajectory.json").write_text(
        trajectory.revalidate().model_dump_json() + "\n"
    )


if __name__ == "__main__":
    from verification2.experiments.collection import Job

    collect(Job.model_validate_json(Path(sys.argv[1]).read_text()))
