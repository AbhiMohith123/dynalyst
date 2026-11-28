"""Scratch folders are pruned by age; saved models never get a doubled extension."""
import json
import os
import time


def test_old_scratch_folders_are_removed_and_recent_ones_kept(tmp_path, monkeypatch):
    from dynalyst import config, housekeeping
    monkeypatch.setattr(housekeeping, "scratch_dirs", lambda: [tmp_path])
    old, new = tmp_path / "old_run", tmp_path / "new_run"
    for d in (old, new):
        d.mkdir()
        (d / "f.txt").write_text("x")
    week_ago = time.time() - 8 * 86400
    for p in (old, old / "f.txt"):
        os.utime(p, (week_ago, week_ago))
    assert housekeeping.prune_scratch(days=7) == 1
    assert not old.exists() and new.exists()
    assert housekeeping.prune_scratch(days=0) == 0       # 0: keep everything


def test_a_name_given_with_its_extension_is_not_doubled():
    from dynalyst import config
    from dynalyst.analysis.save_model import save_sbml
    out = json.loads(save_sbml(str(config.MODELS_DIR / "pitchfork_model.xml"), "test_copy_ext.xml",
                               overwrite=True))
    try:
        assert not out.get("error"), out
        assert out["path"].endswith("test_copy_ext.xml") and ".xml.xml" not in out["path"]
    finally:
        (config.LOCAL_SBML_MODELS_DIR / "test_copy_ext.xml").unlink(missing_ok=True)
