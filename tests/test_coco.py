from pathlib import Path

from core.coco import empty_coco, load_coco, polygon_bbox_area, replace_annotations, save_coco
from engine.types import ClassSpec


def test_coco_roundtrip_keeps_polygons_and_classes(tmp_path: Path) -> None:
    classes = [
        ClassSpec(id=1, name="Loop-A", color="#e63946"),
        ClassSpec(id=2, name="Loop-B", color="#2a9d8f"),
    ]
    coco = empty_coco(classes, description="roundtrip")
    coco["images"].append({"id": 1, "file_name": "a.png", "width": 32, "height": 32})
    replace_annotations(
        coco,
        [
            {
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[1, 1, 10, 1, 10, 8, 1, 8]],
            }
        ],
    )
    path = tmp_path / "annotations.json"
    save_coco(path, coco)
    loaded = load_coco(path)
    assert [item["name"] for item in loaded["categories"]] == ["Loop-A", "Loop-B"]
    polygon = loaded["annotations"][0]["segmentation"][0]
    assert polygon == [1, 1, 10, 1, 10, 8, 1, 8]
    assert loaded["annotations"][0]["category_id"] == 2
    bbox, area = polygon_bbox_area(polygon)
    assert bbox[2] == 9
    assert area > 0
