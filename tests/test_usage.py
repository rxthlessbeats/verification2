from verification2.jev.usage import UsageLedger, format_table, summarize


def test_ledger_roundtrip_and_summary(tmp_path):
    ledger = UsageLedger(tmp_path / "u.jsonl")
    ledger.record("typesafe-ai/jev", "exp-a", 300, 20, 0.0000126, 0.4)
    ledger.record("typesafe-ai/jev", "exp-a", 200, 20, 0.0000084, 0.2)
    ledger.record("typesafe-ai/jev", "exp-b", 0, 0, 0.0, 0.1, status=403)

    total, by_run = summarize(ledger.read(), group_by="run")
    assert total.calls == 3 and total.ok == 2
    assert total.input_tokens == 500
    assert abs(total.cost_usd - 0.000021) < 1e-9
    assert by_run["exp-a"].calls == 2 and by_run["exp-b"].ok == 0
    assert abs(by_run["exp-a"].mean_latency_s - 0.3) < 1e-9

    table = format_table(total, by_run, "run")
    assert "exp-a" in table and "TOTAL" in table and "500" in table


def test_summary_by_day_uses_utc_date(tmp_path):
    ledger = UsageLedger(tmp_path / "u.jsonl")
    ledger.record("m", "r", 10, 1, 0.0, 0.0)
    _, by_day = summarize(ledger.read(), group_by="day")
    assert len(by_day) == 1
    (day,) = by_day
    assert len(day) == 10 and day.count("-") == 2


def test_empty_ledger(tmp_path):
    total, groups = summarize(UsageLedger(tmp_path / "none.jsonl").read())
    assert total.calls == 0 and groups == {}
