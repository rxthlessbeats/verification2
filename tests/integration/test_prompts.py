import json
from unittest.mock import patch

import pytest
import yaml

from qera.jsonl import write_jsonl
from tests.support import sample
from verification2.agents.verifier import digest, main
from verification2.core.prompts import DEFAULT_PROMPTS, load_prompts


@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    ["blank", "missing", "unknown", "wrong_type", "version", "duplicate", "unsafe"],
)
def test_invalid_prompt_yaml_is_rejected_before_outputs_or_calls(tmp_path, case):
    bundle = load_prompts().model_dump()
    if case == "missing":
        del bundle["jev"]["criteria"]["R"]
    elif case == "unknown":
        bundle["verifier"]["instructon"] = "typo"
    elif case == "wrong_type":
        bundle["jev"]["instruction"] = 123
    elif case == "version":
        bundle["version"] = True
    elif case == "blank":
        bundle["verifier"]["instruction"] = " \n"
    contents = yaml.safe_dump(bundle)
    if case == "duplicate":
        contents += "\nversion: 1\n"
    elif case == "unsafe":
        contents = '!!python/object/apply:os.system ["false"]'
    prompt_file = tmp_path / "invalid.yaml"
    prompt_file.write_text(contents)
    output = tmp_path / "run"
    with patch("urllib.request.urlopen") as transport, pytest.raises(ValueError):
        main(
            [
                "verify",
                "--input",
                "missing.jsonl",
                "--output",
                str(output),
                "--prompt-file",
                str(prompt_file),
            ]
        )
    transport.assert_not_called()
    assert not output.exists()


@pytest.mark.integration
def test_jev_uses_yaml_instruction_and_criteria_and_records_resolved_provenance(
    tmp_path,
):
    source = tmp_path / "source.jsonl"
    write_jsonl([sample()], str(source))
    prompt_file = tmp_path / "experiment.yaml"
    bundle = load_prompts().model_dump()
    bundle["jev"]["instruction"] = "Custom classification instruction.\n"
    bundle["jev"]["criteria"]["E"] = "External data: custom criterion."
    prompt_file.write_text(yaml.safe_dump(bundle))
    output = tmp_path / "prepared"
    with pytest.raises(SystemExit) as status:
        main(
            [
                "classify",
                "--input",
                str(source),
                "--output",
                str(output),
                "--prompt-file",
                str(prompt_file),
                "--model",
                "typesafe-ai/jev",
            ]
        )
    assert status.value.code == 0
    lines = (output / "requests.jsonl").read_text().splitlines()
    for request in (json.loads(line)["request"] for line in lines):
        (question,) = request["questions"].values()
        assert question["instructions"]["task"] == bundle["jev"]["instruction"]
        assert question["criteria"] == bundle["jev"]["criteria"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["prompt"] == bundle["jev"]
    assert manifest["prompt_hash"] == digest(bundle["jev"])
    assert manifest["configuration"]["prompt_file"] == str(prompt_file)
    assert load_prompts().model_dump() == yaml.safe_load(DEFAULT_PROMPTS.read_text())
