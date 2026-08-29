import unittest

from dfxlab.schema import anonymize_id, redact


class SchemaTest(unittest.TestCase):
    def test_redacts_nested_request_content(self) -> None:
        payload = {
            "request_id": "request-1",
            "sampling_params": {
                "structured_outputs": {"json": {"private": "schema"}}
            },
            "prompt_token_ids": [1, 2, 3],
            "metrics": {"waiting": 2},
        }
        redacted = redact(payload)
        self.assertEqual(redacted["prompt_token_ids"], "<redacted>")
        self.assertEqual(
            redacted["sampling_params"]["structured_outputs"]["json"],
            "<redacted>",
        )
        self.assertEqual(redacted["metrics"], {"waiting": 2})

    def test_anonymized_ids_are_stable(self) -> None:
        self.assertEqual(anonymize_id("a"), anonymize_id("a"))
        self.assertNotEqual(anonymize_id("a"), anonymize_id("b"))


if __name__ == "__main__":
    unittest.main()

