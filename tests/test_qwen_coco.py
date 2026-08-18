import json

from ellipsoid_steering.qwen_coco import load_split, save_chair_captions, write_coco_split


def test_coco_split_is_deterministic_and_disjoint(tmp_path):
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({"images": [
        {"id": i, "file_name": f"{i}.jpg"} for i in range(20)]}))
    first = write_coco_split(annotations, tmp_path / "a.json", 6, 4, seed=7)
    second = write_coco_split(annotations, tmp_path / "b.json", 6, 4, seed=7)
    assert first["fit"] == second["fit"]
    assert {x["id"] for x in first["fit"]}.isdisjoint(
        {x["id"] for x in first["eval"]})
    assert len(load_split(tmp_path / "a.json", "eval")) == 4


def test_chair_caption_schema(tmp_path):
    output = tmp_path / "captions.json"
    save_chair_captions(output, [{"image_id": "12", "caption": "A dog."}])
    assert json.loads(output.read_text()) == [{"image_id": 12, "caption": "A dog."}]
