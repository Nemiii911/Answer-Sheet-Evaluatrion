"""Offline regression tests for the assignment pipeline."""
import json
import tempfile
import unittest
from pathlib import Path

import pipeline


ROOT = Path(__file__).parent


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = json.loads((ROOT / "answer_key.json").read_text(encoding="utf-8"))
        cls.pages = [
            (ROOT / "sample_transcripts" / "page1.txt").read_text(encoding="utf-8"),
            (ROOT / "sample_transcripts" / "page2.txt").read_text(encoding="utf-8"),
        ]

    def test_spillover_and_numbered_list_are_handled(self):
        segments = pipeline.segment(self.pages, {1, 2, 3, 4})
        self.assertEqual(segments[2]["pages"], {1, 2})
        self.assertTrue(segments[2]["spillover"])
        self.assertIn("Veins normally carry deoxygenated blood", " ".join(segments[2]["lines"]))
        self.assertIn("1. Arteries have high pressure", " ".join(segments[2]["lines"]))
        self.assertTrue(segments[3]["lines"][0].startswith("A body that is resting"))

    def test_markers_are_preserved_for_confidence_and_cleaned_for_judging(self):
        segments = pipeline.segment(self.pages, {1, 2, 3, 4})
        q3 = pipeline.analyse_answer(segments[3]["lines"])
        q4 = pipeline.analyse_answer(segments[4]["lines"])
        self.assertTrue(q3["has_crossed_out"])
        self.assertNotIn("Objects stop without any force", q3["clean_text"])
        self.assertTrue(q4["has_diagram"])
        self.assertEqual(q4["uncertain_count"], 1)

    def test_offline_run_writes_required_output_fields(self):
        rows = pipeline.run(self.pages, self.key, judge_runs=2, baseline=True, allow_bare=False)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all({"question_no", "extracted_answer", "score", "score_vs_key", "confidence", "reason"} <= row.keys() for row in rows))
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            pipeline.write_outputs(rows, temp_dir)
            self.assertTrue((Path(temp_dir) / "results.json").exists())
            self.assertTrue((Path(temp_dir) / "results.csv").exists())

    def test_openai_request_shape_for_text_and_multimodal_input(self):
        class FakeResponses:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return type("Response", (), {"output_text": "ok"})()

        fake_responses = FakeResponses()
        original_client = pipeline._client
        try:
            pipeline._client = type("Client", (), {"responses": fake_responses})()
            self.assertEqual(pipeline.call_llm("system", "prompt"), "ok")
            parts = [{"type": "input_text", "text": "prompt"}]
            self.assertEqual(pipeline.call_llm("system", parts), "ok")
        finally:
            pipeline._client = original_client

        self.assertEqual(fake_responses.calls[0]["input"], "prompt")
        self.assertEqual(fake_responses.calls[1]["input"], [{"role": "user", "content": parts}])
        self.assertEqual(fake_responses.calls[1]["instructions"], "system")

    def test_encode_file_uses_openai_data_url_parts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            image_path = Path(temp_dir) / "page.jpg"
            pdf_path = Path(temp_dir) / "page.pdf"
            image_path.write_bytes(b"image-data")
            pdf_path.write_bytes(b"pdf-data")
            image = pipeline.encode_file(image_path)
            pdf = pipeline.encode_file(pdf_path)

        self.assertEqual(image["type"], "input_image")
        self.assertTrue(image["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(pdf["type"], "input_file")
        self.assertEqual(pdf["filename"], "page.pdf")
        self.assertTrue(pdf["file_data"].startswith("data:application/pdf;base64,"))


if __name__ == "__main__":
    unittest.main()
