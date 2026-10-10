from spatial_scene_bench.generate import build_dataset

from spatial_r1.data import build_examples, split_by_scene


def test_build_examples(tmp_path):
    build_dataset(tmp_path, num_scenes=4, video_frames=12, width=96, height=72, seed=1)
    train, val = build_examples(tmp_path, max_frames=4, val_fraction=0.25)
    assert train and val
    assert {e["id"].split("_")[0] for e in train}.isdisjoint({e["id"].split("_")[0] for e in val})
    for e in train + val:
        n_img = sum(c["type"] == "image" for c in e["prompt"][0]["content"])
        assert n_img == len(e["image_paths"]) <= 4
        assert "<answer>" in e["prompt"][0]["content"][-1]["text"]


def test_split_is_scene_disjoint():
    recs = [{"scene_id": f"s{i % 10}", "id": str(i)} for i in range(100)]
    tr, va = split_by_scene(recs, 0.2)
    assert len(tr) + len(va) == 100
    assert {r["scene_id"] for r in tr}.isdisjoint({r["scene_id"] for r in va})
