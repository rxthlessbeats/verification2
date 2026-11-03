import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import unittest
import tempfile
from pathlib import Path


def raises(error, match=None):
    case = unittest.TestCase()
    return case.assertRaisesRegex(error, match) if match else case.assertRaises(error)

from qera.cli import read_jsonl
from qera.render import build_verifier_input, selector_view
from verification2.pipeline import Backend, choose, demo, main, metrics, parse_response


def test_demo_and_mock_no_label_leak(tmp_path):
    source = tmp_path / 'sample.jsonl'
    demo(source)
    output = tmp_path / 'mock'
    with raises(SystemExit) as result:
        main(['run', '--input', str(source), '--output', str(output), '--backend', 'mock'])
    assert result.exception.code == 0
    requests = (output / 'requests.jsonl').read_text()
    assert 'DO_NOT_SEND_HIDDEN' not in requests
    assert 'label_correct' not in requests and 'hidden' not in requests
    results = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    assert len(results) == 16 and all(r['status'] == 'mock' for r in results)
    assert all(s['balanced_accuracy'] is None for s in metrics(results, ('QERA', 'QA', 'QEA', 'QEsA')))


def test_selection_budget_and_file_errors(tmp_path):
    source = tmp_path / 'sample.jsonl'
    demo(source)
    trajectory = next(read_jsonl([source]))
    view = selector_view(trajectory)
    for method in ('random', 'recency', 'relevance'):
        assert choose(view, method, 0, 42, {}) == []
        chosen = choose(view, method, 20, 42, {})
        assert chosen == choose(view, method, 20, 42, {})
        build_verifier_input(trajectory, 'QEsA', selected_e=chosen, budget=20)
    with raises(ValueError, match='Missing'):
        choose(view, 'file', 100, 0, {})
    with raises(ValueError, match='non-E'):
        choose(view, 'file', 100, 0, {trajectory.traj_id: [0]})
    with raises(ValueError, match='budget'):
        choose(view, 'file', 0, 0, {trajectory.traj_id: [1]})


def test_response_parser():
    assert parse_response('```json\n{"verdict":"incorrect","reason":"bad"}\n```')['verdict'] == 'incorrect'
    for bad in ['correct', '{"verdict":true,"reason":"x"}', '{"verdict":"correct"}']:
        with raises((ValueError, TypeError)):
            parse_response(bad)


def test_paired_metrics_exclude_failed_pair():
    rows = []
    for i, label in enumerate([True, False, True]):
        for condition in ('QA', 'QEA'):
            rows.append(dict(benchmark='b', traj_id=str(i), condition=condition, label_correct=label,
                             status='error' if i == 2 and condition == 'QA' else 'ok',
                             verdict='correct' if label else 'incorrect'))
    summaries = metrics(rows, ('QA', 'QEA'))
    assert all(s['paired_labelled_n'] == 2 for s in summaries)
    assert all(s['balanced_accuracy'] == 1 for s in summaries)


def test_http_wire_contract():
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            response = {'choices': [{'message': {'content': '{"verdict":"incorrect","reason":"test"}'}}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        request = {'model': 'local-test', 'messages': [{'role': 'user', 'content': 'test'}]}
        response = Backend('http', 'http://127.0.0.1:%d/v1/chat/completions' % server.server_port).call(request)
        assert response['verdict'] == 'incorrect'
        assert response['usage']['prompt_tokens'] == 10
        assert captured == [request]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_existing_output_not_overwritten(tmp_path):
    source = tmp_path / 'sample.jsonl'
    demo(source)
    output = tmp_path / 'existing'
    output.mkdir()
    with raises(ValueError, match='already exists'):
        main(['run', '--input', str(source), '--output', str(output)])


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, function in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(function):
            def run_case(fn=function):
                with tempfile.TemporaryDirectory() as directory:
                    if fn.__code__.co_argcount:
                        fn(Path(directory))
                    else:
                        fn()
            suite.addTest(unittest.FunctionTestCase(run_case, description=name))
    return suite


if __name__ == "__main__":
    unittest.main()
