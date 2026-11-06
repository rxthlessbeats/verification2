"""Freeze training-selected settings before any held-out pipeline stage."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

import qera

from ..benchmarks.setup import ROOT
from ..core.config import Endpoint, Name, PositiveInt, Settings
from ..core.prompts import load_prompts
from ..core.registry import max_output_tokens, resolve
from ..data.split import BENCHMARKS, digest, load_split, split_id


def code_hash():
    value = hashlib.sha256()
    for package in sorted((Path(qera.__file__).parent, Path(__file__).parents[1])):
        for path in sorted(package.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                value.update(
                    (
                        "src/" + package.name + "/" + str(path.relative_to(package))
                    ).encode()
                )
                value.update(path.read_bytes())
    if (ROOT / "uv.lock").is_file():
        value.update((ROOT / "uv.lock").read_bytes())
    return value.hexdigest()


def runtime_hashes(required=False, benchmarks=BENCHMARKS):
    from ..benchmarks.doctor import LOCKS, NEEDS

    result = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        if (ROOT / p).is_file()
        else "missing"
        for p in LOCKS.values()
    }
    needed = {name for benchmark in benchmarks for name in NEEDS[benchmark]}
    if required and any(
        result[path] == "missing" for name, path in LOCKS.items() if name in needed
    ):
        raise ValueError(
            "Native runtime locks are incomplete for the selected benchmarks; run make setup before freezing or using held-out data"
        )
    return result


def solver_prompt_hash(prompts):
    return digest({b: prompts.solver(b) for b in BENCHMARKS})


def check_pilot(path, splits):
    if path is None or not Path(path).is_file():
        raise ValueError(
            "Formal collection requires --pilot-report from a successful pilot step"
        )
    pilot = json.loads(Path(path).read_text())
    if (
        pilot.get("status") != "passed"
        or pilot.get("split_id") != split_id(splits)
        or pilot.get("source_hash") != code_hash()
        or pilot.get("runtime_hashes") != runtime_hashes()
        or set(pilot.get("readable_reasoning", {})) != {s.benchmark for s in splits}
        or not all(pilot["readable_reasoning"].values())
    ):
        raise ValueError(
            "The reasoning-aware pilot must pass for these benchmarks with the current source and runtimes"
        )


class Protocol(Settings):
    version: Literal[1] = 1
    split_id: Name
    source_hash: Name
    solver_model: Name
    solver_endpoint: Endpoint
    solver_reasoning_effort: Name | None = None
    solver_prompt_hash: Name
    runtime_hashes: dict[str, str]
    jev_model: Name
    jev_endpoint: Endpoint
    jev_prompt_hashes: dict[str, str]
    verifier_model: Name
    verifier_endpoint: Endpoint
    verifier_reasoning_effort: Name | None = None
    verifier_prompt_hash: Name
    verifier_max_tokens: PositiveInt | None = None
    temperature: Literal[0] = 0
    conditions: list[Literal["QA", "QEA", "QRA", "QERA"]] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_conditions(self):
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("Duplicate held-out conditions")
        return self


def check_heldout(splits, partition, path, solver, prompts):
    if partition != "test":
        return None
    if path is None:
        raise ValueError("Held-out processing requires --protocol from the freeze step")
    protocol = Protocol.model_validate_json(Path(path).read_text())
    if protocol.split_id != split_id(splits) or protocol.source_hash != code_hash():
        raise ValueError(
            "Held-out protocol does not match the frozen split/application source"
        )
    if (
        protocol.solver_model != solver.model
        or protocol.solver_endpoint != solver.endpoint
        or protocol.solver_reasoning_effort != solver.reasoning_effort
        or protocol.solver_prompt_hash != solver_prompt_hash(prompts)
    ):
        raise ValueError(
            "Solver model, reasoning effort or prompt differs from the held-out protocol"
        )
    if protocol.runtime_hashes != runtime_hashes(
        required=True, benchmarks=[s.benchmark for s in splits]
    ):
        raise ValueError(
            "Native dependency locks differ from the frozen held-out protocol"
        )
    return protocol


def check_stage(trajectories, args, prompts, stage):
    heldout = [t for t in trajectories if t.split == "test" and t.meta.get("split_id")]
    if not heldout:
        return
    if args.protocol is None:
        raise ValueError("Held-out processing requires --protocol from the freeze step")
    protocol = Protocol.model_validate_json(args.protocol.read_text())
    if protocol.source_hash != code_hash() or any(
        t.meta["split_id"] != protocol.split_id for t in heldout
    ):
        raise ValueError("Held-out protocol/source/split mismatch")
    if protocol.runtime_hashes != runtime_hashes(
        required=True, benchmarks={t.benchmark for t in heldout}
    ):
        raise ValueError(
            "Native dependency locks differ from the frozen held-out protocol"
        )
    if args.model != getattr(protocol, stage + "_model") or args.endpoint != getattr(
        protocol, stage + "_endpoint"
    ):
        raise ValueError("Model/endpoint differs from the frozen held-out protocol")
    if stage == "jev":
        if any(
            digest(prompts.classification(t.benchmark).model_dump(mode="json"))
            != protocol.jev_prompt_hashes.get(t.benchmark)
            for t in heldout
        ):
            raise ValueError("Jev prompt differs from the frozen held-out protocol")
    elif (
        digest(prompts.verifier.model_dump(mode="json"))
        != protocol.verifier_prompt_hash
        or list(args.conditions) != protocol.conditions
    ):
        raise ValueError(
            "Verifier prompt/conditions differ from the frozen held-out protocol"
        )
    if (
        stage == "verifier"
        and args.reasoning_effort != protocol.verifier_reasoning_effort
    ):
        raise ValueError(
            "Verifier reasoning effort differs from the frozen held-out protocol"
        )
    if (
        stage == "verifier"
        and args.max_tokens is not None
        and args.max_tokens != protocol.verifier_max_tokens
    ):
        raise ValueError(
            "Verifier output maximum differs from the frozen held-out protocol"
        )
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--conditions", nargs="+", default=["QA", "QEA", "QRA", "QERA"])
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args(argv)
    splits = [load_split(b) for b in args.benchmarks]
    prompts = load_prompts(args.prompt_file)
    locks = runtime_hashes(required=True, benchmarks=args.benchmarks)
    solver = resolve("solver", os.environ.get("SOLVER_MODEL"))
    jev = resolve("jev", os.environ.get("JEV_MODEL"))
    verifier = resolve("verifier", os.environ.get("VERIFIER_MODEL"))
    maximum = args.max_tokens
    if maximum is None and verifier.api == "messages":
        maximum = max_output_tokens(verifier.model)
    protocol = Protocol(
        split_id=split_id(splits),
        source_hash=code_hash(),
        solver_model=solver.model,
        solver_endpoint=solver.endpoint,
        solver_reasoning_effort=solver.reasoning_effort,
        solver_prompt_hash=solver_prompt_hash(prompts),
        runtime_hashes=locks,
        jev_model=jev.model,
        jev_endpoint=jev.endpoint,
        jev_prompt_hashes={
            b: digest(prompts.classification(b).model_dump(mode="json"))
            for b in BENCHMARKS
        },
        verifier_model=verifier.model,
        verifier_endpoint=verifier.endpoint,
        verifier_reasoning_effort=verifier.reasoning_effort,
        verifier_max_tokens=maximum,
        verifier_prompt_hash=digest(prompts.verifier.model_dump(mode="json")),
        conditions=args.conditions,
    )
    if args.output.exists():
        if Protocol.model_validate_json(args.output.read_text()) != protocol:
            raise ValueError("Existing protocol differs; it was preserved")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as target:
            target.write(protocol.model_dump_json(indent=2) + "\n")
    print("Frozen held-out protocol:", args.output)


if __name__ == "__main__":
    main()
