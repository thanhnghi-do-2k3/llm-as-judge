import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NotebookRequirementsTests(unittest.TestCase):
    def test_notebook_requirements_do_not_replace_scientific_stack(self):
        requirements = (ROOT / "requirements-notebook.txt").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("-r requirements-core.txt", requirements)
        self.assertIsNone(
            re.search(r"(?m)^(numpy|pandas|scipy)==", requirements)
        )


if __name__ == "__main__":
    unittest.main()
