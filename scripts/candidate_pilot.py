"""Evaluate one answer-blind, reproducible candidate task through the project's four views."""

import argparse
import ast
import hashlib
import json
import operator
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from qera.adapters.text_env import convert_messages
from qera.jsonl import read as read_rows
from verification2.benchmarks.setup import ROOT, _download, file_hash
from verification2.benchmarks.text_loop import EpisodeLimitReached, run, tool
from verification2.core.registry import resolve

DATA = ROOT / "raw/candidate-pilots"
GSM = "https://huggingface.co/datasets/qintongli/GSM-Plus/resolve/3b708db57b96a16e8e3368ed2956990c0809440e/data/test-00000-of-00001.jsonl"
ASSISTANT = "https://huggingface.co/datasets/AssistantBench/AssistantBench/resolve/482cbbc0400f6d048438c4021727f21a10cbff49/assistant_bench_v1.0_dev.jsonl"
GAIA_REV = "682dd723ee1e1697e00360edccf2366dc8418dd9"
GAIA_SCORER = "https://huggingface.co/spaces/gaia-benchmark/leaderboard/raw/9f133d71362e77b3539f1514f31b9c101a545fec/scorer.py"
ASSISTANT_WHEEL = "https://files.pythonhosted.org/packages/c4/07/09199cc82b62562aea118bbbfc2ed8a2a023ee79fc2b0f93d747ed86bfb8/browsergym_assistantbench-0.14.3-py3-none-any.whl"
VIEWS = ["QA", "QEA", "QRA", "QERA"]


def artifact(name, url, digest):
    path = DATA / name
    _download(url, path)
    if file_hash(path) != digest:
        raise ValueError(f"Unexpected release checksum for {name}")
    return path


def case(benchmark, gaia_file, paper_example=False):
    if paper_example:
        if benchmark != "gaia" or gaia_file:
            raise ValueError("The paper example is a separate GAIA demonstration")
        # Public worked example in Table 1, https://arxiv.org/pdf/2311.12983#page=20.
        return (
            "paper-table-1",
            "According to the NIH trial registry, how many participants actually enrolled in the January–May 2018 study of H. pylori in patients with acne vulgaris?",
            "90",
        )
    if benchmark == "gaia":
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = gaia_file or hf_hub_download(
            "gaia-benchmark/GAIA",
            "2023/validation/metadata.parquet",
            repo_type="dataset",
            revision=GAIA_REV,
        )
        rows = [r for r in pq.read_table(path).to_pylist() if not r["file_name"]]
        field, answer = "Question", "Final answer"
    else:
        if benchmark == "assistantbench":
            path = artifact(
                "assistantbench.jsonl",
                ASSISTANT,
                "c1abd710b99d6f810e2ce95cb6c43f31828dacd778c9e4d9dd4ddad2eaee6ac4",
            )
            field = "task"
        else:
            path = artifact(
                "gsm_plus.jsonl",
                GSM,
                "885c279bfa9cde767ea23f79fd462d566fa0dbf6af9f2dcf30c75aa94f3d4041",
            )
            field = "question"
        rows = read_rows(path)
        answer = "answer"
    row = min(rows, key=lambda r: hashlib.sha256(("42:" + r[field]).encode()).digest())
    ident = (
        row.get("id")
        or row.get("task_id")
        or hashlib.sha256(row[field].encode()).hexdigest()
    )
    return ident, row[field], row[answer]


def calculator(expression):
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    if len(expression) > 256:
        return "Expression too long."

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", literal):
                raise ValueError
            return Fraction(literal)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            number = value(node.operand)
            return -number if isinstance(node.op, ast.USub) else number
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.left), value(node.right))
        raise ValueError

    try:
        return str(value(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ArithmeticError, RecursionError):
        return "Use numbers, parentheses and + - * /."


def search(query, cache):
    """Cache search results. A local operator can relay public search when no API key is set."""
    key = hashlib.sha256(query.encode()).hexdigest()
    directory = cache.parent / "search"
    directory.mkdir(exist_ok=True)
    path = directory / (key + ".json")
    if not path.exists():
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if api_key:
            url = (
                "https://api.search.brave.com/res/v1/web/search?"
                + urllib.parse.urlencode({"q": query})
            )
            request = urllib.request.Request(
                url,
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as reply:
                hits = json.load(reply).get("web", {}).get("results", [])
            path.write_text(
                json.dumps(
                    [
                        {
                            "title": h["title"],
                            "url": h["url"],
                            "snippet": h["description"],
                        }
                        for h in hits
                    ]
                )
            )
        else:
            request_path = directory / (key + ".request")
            request_path.write_text(query)
            print("Search requested:", query, "->", path, flush=True)
            deadline = time.monotonic() + 600
            while not path.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("No search receipt received")
                time.sleep(1)
    hits = json.loads(path.read_text())
    if not isinstance(hits, list) or any(
        not all(isinstance(h.get(k), str) for k in ("title", "url", "snippet"))
        for h in hits
    ):
        raise ValueError("Invalid search receipt")
    results = [
        h
        for h in hits
        if not any(
            marker in (h["title"] + " " + h["url"] + " " + h["snippet"]).lower()
            for marker in (
                "huggingface",
                "github.com",
                "assistantbench",
                "gaia-benchmark",
            )
        )
    ]
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


def scorer(benchmark):
    if benchmark == "gsm_plus":

        def numeric(pred, gold):
            try:
                return float(Fraction(pred.strip().replace(",", "")) == Fraction(gold))
            except (ValueError, ZeroDivisionError):
                return 0.0

        return numeric
    if benchmark == "gaia":
        artifact(
            "gaia_scorer.py",
            GAIA_SCORER,
            "0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2",
        )
        sys.path.insert(0, str(DATA))
        from gaia_scorer import question_scorer

        return question_scorer
    wheel = artifact(
        "assistantbench.whl",
        ASSISTANT_WHEEL,
        "3d489b39eb7889ec0cfc7a8c0c097c549fc4e4b6bf7ca4df256a91c8d6e25713",
    )
    # Import only the released evaluator. The package's top-level init launches BrowserGym.
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith(
                "browsergym/assistantbench/evaluation/"
            ) and name.endswith(".py"):
                archive.extract(name, DATA)
    sys.path.insert(0, str(DATA / "browsergym/assistantbench"))
    from evaluation.evaluator import question_scorer

    return lambda pred, gold: question_scorer(pred, gold)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("gsm_plus", "gaia", "assistantbench"))
    parser.add_argument(
        "--gaia-file", type=Path, help="Official local validation metadata.parquet"
    )
    parser.add_argument(
        "--gaia-paper-example",
        action="store_true",
        help="Use the paper's public worked example, separately from the gated validation pilot",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name, root = args.benchmark, args.output
    ident, question, gold = case(name, args.gaia_file, args.gaia_paper_example)
    grade = scorer(name)
    solver = resolve("solver", os.environ.get("SOLVER_MODEL"))
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        "benchmark": name,
        "task_id": ident,
        "source": "paper-table-1"
        if args.gaia_paper_example
        else "validation"
        if name == "gaia"
        else "dev"
        if name == "assistantbench"
        else "v1-test",
        "views": VIEWS,
        "models": {
            role: {
                "model": config.model,
                "effort": config.reasoning_effort,
                "endpoint": config.endpoint,
            }
            for role in ("solver", "jev", "verifier")
            for config in [resolve(role, os.environ.get(role.upper() + "_MODEL"))]
        },
    }
    settings_path = root / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Use a fresh pilot output directory for changed settings")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    source = root / "trajectory.jsonl"
    if not source.exists():
        os.environ["SOLVER_API_KEY"] = os.environ[solver.spec.api_key_env]
        job = SimpleNamespace(
            model=solver.model,
            endpoint=solver.endpoint,
            reasoning_effort=solver.reasoning_effort,
            output=root,
            cache_dir=cache,
            solver_instruction=None,
        )
        web = name != "gsm_plus"
        system = "Answer the question. Use calculator for arithmetic. "
        if web:
            system += "Use search for public web evidence. It returns search result snippets and URLs. Avoid benchmark datasets and answer keys. "
        system += "Finish by calling submit with just the final answer. If you cannot determine it, submit unknown."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        functions = [tool("calculator", "Evaluate an arithmetic expression.")]
        if web:
            functions.append(tool("search", "Search the public web by keyword."))
        functions.append(tool("submit", "Submit the final answer."))
        submitted = None

        def handle(action, command):
            nonlocal submitted
            if action == "submit" and command.strip():
                submitted = command.strip()
                return "Answer recorded.", True
            if action == "calculator":
                return calculator(command), False
            if action == "search" and web:
                return search(command, cache), False
            return "Use one of the available tools.", False

        try:
            run(job, messages, functions, handle, limit=16)
        except EpisodeLimitReached:
            messages.append(
                {
                    "role": "user",
                    "content": "Research turns are over. Submit your best supported answer now, or submit unknown.",
                }
            )
            run(job, messages, [functions[-1]], handle, limit=1)
        if submitted is None:
            raise RuntimeError(
                "Pilot tool budget exhausted without submission. Episode is unscored."
            )
        score = float(grade(submitted, gold))
        trajectory = convert_messages(
            messages,
            benchmark=name,
            task_domain=None,
            task_id=ident,
            model=solver.model,
            run_id=root.name,
            correct=score == 1.0,
            score=score,
            detail={"full_credit": score == 1.0},
            agent="candidate_pilot",
            label_source="exact_match" if name != "assistantbench" else "unit_tests",
        )
        trajectory.split = "pilot"
        source.write_text(trajectory.revalidate().model_dump_json() + "\n")
    else:
        from qera.schema import Trajectory

        saved = Trajectory.model_validate_json(source.read_text())
        if (saved.benchmark, saved.task_id, saved.solver.model) != (
            name,
            ident,
            solver.model,
        ):
            raise ValueError("Output contains a different episode")
    for stage, filename, extra in [
        ("classify", "trajectories.jsonl", []),
        ("verify", "results.jsonl", ["--conditions", *VIEWS, "--jobs", "4"]),
    ]:
        out = root / stage
        attempt = 0
        while out.exists():
            artifact_path = out / filename
            if artifact_path.is_file():
                if stage == "classify":
                    ready = json.loads((out / "manifest.json").read_text()).get(
                        "complete"
                    )
                else:
                    rows = list(read_rows(artifact_path))
                    ready = len(rows) == len(VIEWS) and {
                        r["condition"] for r in rows if r["status"] == "ok"
                    } == set(VIEWS)
                if ready:
                    break
            attempt += 1
            out = root / f"{stage}.attempt{attempt}"
        if not out.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "verification2.agents.verifier",
                    stage,
                    "--backend",
                    "http",
                    "--input",
                    str(source),
                    "--output",
                    str(out),
                    "--cache-dir",
                    str(cache),
                    *extra,
                ],
                check=True,
            )
        source = out / filename
    results = list(read_rows(source))
    if {r["condition"] for r in results if r["status"] == "ok"} != set(VIEWS):
        raise ValueError("Pilot is missing verifier verdicts")
    outcome = json.loads((root / "trajectory.jsonl").read_text())["label"]
    (root / "report.json").write_text(
        json.dumps(
            {
                **settings,
                "partition": "pilot",
                "tasks": 1,
                "score": outcome["score"],
                "full_credit": outcome["correct"],
                "verdicts": {r["condition"]: r["verdict"] for r in results},
            },
            indent=2,
        )
        + "\n"
    )
    print(name, [(r["condition"], r["verdict"]) for r in results], flush=True)


if __name__ == "__main__":
    main()
