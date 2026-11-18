import json

import pandas as pd
import pytest

from visionops.config import DATASET_DIR, REPORTS_DIR
from visionops.data.download import extract_zip
from visionops.data.prepare import prepare, read_split, source_group
from visionops.data.validate import validate


def test_source_group():
    assert source_group("x/board_01_jpg.rf.abc123.jpg") == "board_01_jpg"
    assert source_group("x/plain.png") == "plain"


def test_prepare_carves_grouped_test_split_without_leakage(roboflow_export):
    m = prepare(roboflow_export, DATASET_DIR, seed=42, test_frac=0.2)
    assert m["method"]["test"].startswith("carved") and m["method"]["val"] == "provided"
    groups = {s: {source_group(p) for p in read_split(s)} for s in ("train", "val", "test")}
    assert not groups["train"] & groups["test"]
    assert m["counts"]["test"] == 8  # 4 of 20 source images x 2 augmented copies
    # deterministic
    m2 = prepare(roboflow_export, DATASET_DIR, seed=42, test_frac=0.2)
    assert m2["counts"] == m["counts"]

    rep = validate()
    assert not rep["fatal"]
    assert rep["leakage"]["source_groups_in_multiple_splits"] == 0
    assert rep["splits"]["train"]["instances"] == m["counts"]["train"]


def test_validate_flags_bad_class_id(roboflow_export):
    prepare(roboflow_export, DATASET_DIR, seed=1)
    lbl = read_split("train")[0].parent.parent / "labels"
    first = sorted(lbl.glob("*.txt"))[0]
    first.write_text("7 0.5 0.5 0.1 0.1\n")
    rep = validate()
    assert any("invalid_class_id" in f for f in rep["fatal"])


def test_extract_zip_blocks_traversal(tmp_path):
    import zipfile

    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        extract_zip(z, tmp_path / "out")


def test_build_variants_matched_removal(roboflow_export):
    from visionops.experiment import build_variants

    prepare(roboflow_export, DATASET_DIR, seed=42)
    train = [str(p) for p in read_split("train")]
    q = pd.DataFrame({"image": train[:6], "score": [9, 8, 7, 6, 5, 4],
                      "possible_annotation_issue": [True, True, False, True, True, True]})
    qpath = REPORTS_DIR / "review" / "review_queue.csv"
    qpath.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(qpath, index=False)
    man = build_variants(frac=0.1, seed=0, source="queue", queue_csv=qpath)
    cur, rnd = man["variants"]["curated"], man["variants"]["random_control"]
    assert cur["removed_images"] > 0
    assert rnd["removed_images"] >= cur["removed_images"]
    assert train[2] not in cur["removed"] or source_group(train[2]) in {source_group(x) for x in cur["removed"]}
    # whole source groups removed together
    rem_groups = {source_group(x) for x in cur["removed"]}
    assert all(source_group(p) not in rem_groups for p in
               (DATASET_DIR / "experiments" / "curated" / "splits" / "train.txt").read_text().split())
    # val/test untouched: experiment yaml points at the baseline lists
    import yaml

    y = yaml.safe_load(open(cur["dataset"]))
    base = yaml.safe_load(open(DATASET_DIR / "dataset.yaml"))
    assert y["test"] == base["test"] and y["val"] == base["val"] and y["train"] != base["train"]
    assert json.loads((REPORTS_DIR / "experiments" / "manifest.json").read_text())["source"] == "queue"


def test_report_without_results_says_not_run():
    from visionops.report import build

    text = build()
    assert "Not run yet." in text


def test_merge_val_into_test(roboflow_export):
    # add a provided test split, then merge provided valid+test into the held-out test set
    from conftest import make_image, write_label

    root = roboflow_export / "export"
    for i in range(3):
        name = f"test{i:03d}_jpg.rf.ffff.jpg"
        make_image(root / "test" / "images" / name, seed=900 + i)
        write_label(root / "test" / "labels" / name.replace(".jpg", ".txt"), [[0, 0.5, 0.5, 0.2, 0.2]])
    m = prepare(roboflow_export, DATASET_DIR, seed=42, val_frac=0.1, merge_val_into_test=True)
    assert m["counts"]["test"] == 3 + 4  # provided test + provided valid
    assert m["method"]["val"].startswith("carved")
    groups = {s: {source_group(p) for p in read_split(s)} for s in ("train", "val", "test")}
    assert not groups["train"] & groups["val"] and not groups["train"] & groups["test"]
    assert all("valid" in str(p) or "test" in str(p) for p in read_split("test"))


def test_materialize_matches_split_lists(roboflow_export, tmp_path):
    import yaml

    from visionops.data.materialize import materialize

    prepare(roboflow_export, DATASET_DIR, seed=42)
    y = materialize(tmp_path / "dirs")
    cfg = yaml.safe_load(open(y))
    for split in ("train", "val", "test"):
        imgs = sorted(p.name for p in (tmp_path / "dirs" / split / "images").iterdir())
        assert imgs == sorted(p.name for p in read_split(split))
        assert len(list((tmp_path / "dirs" / split / "labels").iterdir())) == len(imgs)
    assert cfg["test"] == "test/images" and len(cfg["names"]) == 2
