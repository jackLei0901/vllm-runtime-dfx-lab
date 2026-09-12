import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "results" / "organic-hang-20260912" / "verify_published_evidence.py"


class PublishedOrganicEvidenceTest(unittest.TestCase):
    def test_derived_evidence_verifies(self) -> None:
        spec = importlib.util.spec_from_file_location("published_evidence", VERIFIER)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load published evidence verifier")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(0, module.main())


if __name__ == "__main__":
    unittest.main()
