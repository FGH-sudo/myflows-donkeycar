"""Archived experiment configs resolve their data paths in any checkout."""

import unittest

from benchmark.config_paths import (
    REPO_ROOT,
    relative_config_paths,
    repo_path,
    repo_relative,
    resolve_config_paths,
)


class ConfigPathTests(unittest.TestCase):
    def test_paths_from_another_checkout_are_rerooted(self):
        expected = str(REPO_ROOT / ".codex" / "distributed-data-v1" / "donkey_cnn")
        for recorded in ("D:\\elsewhere\\testmyflow\\.codex\\distributed-data-v1\\donkey_cnn",
                         "/mnt/d/elsewhere/testmyflow/.codex/distributed-data-v1/donkey_cnn",
                         ".codex/distributed-data-v1/donkey_cnn"):
            self.assertEqual(repo_path(recorded), expected, recorded)

    def test_existing_absolute_paths_are_kept(self):
        self.assertEqual(repo_path(str(REPO_ROOT / "README.md")), str(REPO_ROOT / "README.md"))

    def test_only_file_entries_change(self):
        config = {"prepared_dir": "D:\\x\\.codex\\data", "bn_statistics": None, "learning_rate": 0.1,
                  "task": "docs"}
        resolved = resolve_config_paths(config)
        self.assertEqual(resolved["prepared_dir"], str(REPO_ROOT / ".codex" / "data"))
        self.assertEqual({k: v for k, v in resolved.items() if k != "prepared_dir"},
                         {"bn_statistics": None, "learning_rate": 0.1, "task": "docs"})
        self.assertEqual(relative_config_paths(resolved)["prepared_dir"], ".codex/data")
        self.assertEqual(repo_relative(REPO_ROOT / "docs" / "a.json"), "docs/a.json")


if __name__ == "__main__":
    unittest.main()
