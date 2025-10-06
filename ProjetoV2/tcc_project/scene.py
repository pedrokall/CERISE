from typing import List, Optional

import sionna.rt as srt
from sionna.rt import Camera, ITURadioMaterial, SceneObject


def load_scene_with_options(scene_path: str,
                            merge_shapes: bool = True,
                            merge_shapes_exclude_regex: Optional[str] = None,
                            preview: bool = False,
                            camera_position = (484.48, -212.68, 328.85),
                            camera_look_at = (83.83, -94.6, -0.0667)) -> srt.Scene:
    scene = srt.load_scene(scene_path,
                           merge_shapes=merge_shapes,
                           merge_shapes_exclude_regex=merge_shapes_exclude_regex)
    cam = Camera(position=list(camera_position), look_at=list(camera_look_at))
    if preview:
        scene.preview()
    else:
        scene.render(camera=cam)
    return scene


def create_car_objects(num_cars: int,
                       material: Optional[ITURadioMaterial] = None) -> List[SceneObject]:
    if material is None:
        material = srt.ITURadioMaterial("car-material", "metal", thickness=0.01, color=(0.8, 0.1, 0.1))
    cars = [SceneObject(fname=srt.scene.low_poly_car, name=f"car-{i}", radio_material=material)
            for i in range(num_cars)]
    return cars


def add_objects(scene: srt.Scene, objects: List[SceneObject]) -> None:
    scene.edit(add=objects)


def remove_objects(scene: srt.Scene, names: List[str]) -> None:
    for n in names:
        scene.remove(n)
