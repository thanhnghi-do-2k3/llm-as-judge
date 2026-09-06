import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vn_meta_judge.data.loaders import save_jsonl
from vn_meta_judge.generation.api_runner import generate_damage_with_backend
from vn_meta_judge.generation.deepseek import call_deepseek
from vn_meta_judge.generation.gemini import generate_damage as generate_gemini_damage
from vn_meta_judge.generation.gemini import parse_api_keys, resolve_api_keys


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "Bản hư hại"}}]}).encode("utf-8")


class GenerationBackendTests(unittest.TestCase):
    def test_gemini_key_pool_parses_json_delimiters_and_deduplicates(self):
        self.assertEqual(parse_api_keys('["key-a", "key-b", "key-a"]'), ["key-a", "key-b"])
        self.assertEqual(parse_api_keys("key-a,key-b\nkey-c;key-a"), ["key-a", "key-b", "key-c"])

        with patch("vn_meta_judge.generation.gemini.load_local_env"), patch.dict(
            "os.environ",
            {"GEMINI_API_KEYS": "env-a,env-b", "GEMINI_API_KEY": "env-a"},
            clear=True,
        ):
            self.assertEqual(
                resolve_api_keys(api_key="explicit", api_keys=["pool-a", "explicit"]),
                ["explicit", "pool-a", "env-a", "env-b"],
            )

    def test_gemini_key_pool_round_robins_without_exposing_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "references.jsonl"
            output_path = root / "gemini.jsonl"
            save_jsonl([{"id": "vi-1", "source_zh": "中文", "reference_vi": "Câu tiếng Việt"}], input_path)
            used_keys = []

            def fake_call(messages, model, api_key, max_output_tokens=400):
                used_keys.append(api_key)
                return "Bản hư hại"

            with patch("vn_meta_judge.generation.gemini.load_local_env"), patch.dict(
                "os.environ", {}, clear=True
            ), patch("vn_meta_judge.generation.gemini.call_gemini", side_effect=fake_call):
                result = generate_gemini_damage(
                    input_path,
                    output_path,
                    "gemini-test",
                    "zero_shot",
                    api_keys=["key-a", "key-b"],
                )

            self.assertEqual(result["api_key_count"], 2)
            self.assertEqual(used_keys, ["key-a", "key-b", "key-a", "key-b", "key-a", "key-b"])
            self.assertNotIn("key-a", json.dumps(result))
            self.assertNotIn("key-b", json.dumps(result))

    def test_deepseek_adapter_uses_openai_compatible_contract(self):
        with patch("vn_meta_judge.generation.deepseek.urllib.request.urlopen", return_value=_FakeResponse()) as open_url:
            result = call_deepseek(
                [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
                "deepseek-chat",
                "test-key",
                base_url="https://example.test/",
                max_output_tokens=123,
            )
        self.assertEqual(result, "Bản hư hại")
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "deepseek-chat")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 123)

    def test_runner_persists_backend_and_does_not_mix_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "references.jsonl"
            output_path = root / "gemini.jsonl"
            save_jsonl([{"id": "vi-1", "source_zh": "中文", "reference_vi": "Câu tiếng Việt"}], input_path)

            def fake_caller(messages, model, max_tokens):
                return f"{model}: {messages[-1]['content'].split('Mức damage: ')[1].splitlines()[0]}"

            result = generate_damage_with_backend(
                input_path,
                output_path,
                "gemini-2.5-flash-lite",
                "zero_shot",
                fake_caller,
                backend_name="gemini",
            )
            self.assertEqual(result["rows_written"], 6)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["backend"] for row in rows}, {"gemini"})

            with self.assertRaisesRegex(RuntimeError, "Refusing to mix backends"):
                generate_damage_with_backend(
                    input_path,
                    output_path,
                    "deepseek-chat",
                    "zero_shot",
                    fake_caller,
                    backend_name="deepseek",
                )

            deepseek_result = generate_damage_with_backend(
                input_path,
                root / "deepseek.jsonl",
                "deepseek-chat",
                "zero_shot",
                fake_caller,
                backend_name="deepseek",
            )
            self.assertEqual(deepseek_result["rows_written"], 6)
            deepseek_rows = [json.loads(line) for line in (root / "deepseek.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["backend"] for row in deepseek_rows}, {"deepseek"})


if __name__ == "__main__":
    unittest.main()
