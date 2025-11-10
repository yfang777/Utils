#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render a MuJoCo scene from a newly-added camera.
"""

import argparse
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import os, sys
import mujoco
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_CAMERA_SNIPPET = """
<camera name="left_camera"
        pos="0.8 -0.4 0.4"
        mode="fixed"
        focal="1 1"
        resolution="640 480"
        sensorsize="1 1"
        xyaxes="0.5547002 0.83205029 -0. -0.40360368 0.26906912 0.87447463"/>
""".strip()


def parse_resolution_from_camera(cam_elem):

    res = cam_elem.attrib.get("resolution", "").strip()
    parts = res.split()
    w = int(float(parts[0]))
    h = int(float(parts[1]))
    return w, h

def expand_asset_paths(xml_path: Path, xml_str: str) -> str:
    """
    Make all asset file paths absolute, using the directory of `xml_path` as base.
    Updates <mesh file=...>, <texture file=...>, <hfield file=...>, and <include file=...>.
    Returns the modified XML as a Unicode string.
    """
    base_dir = xml_path.parent.resolve()

    # Parse the in-memory XML
    root = ET.fromstring(xml_str)

    # Helper: namespace-agnostic tag check
    def tag_is(elem, name: str) -> bool:
        return elem.tag == name or elem.tag.endswith("}" + name)

    # Walk all elements; update any supported tag that has a 'file' attribute
    for elem in root.iter():
        if tag_is(elem, "mesh") or tag_is(elem, "texture") or tag_is(elem, "hfield") or tag_is(elem, "include"):
            f = elem.attrib.get("file")
            if not f:
                continue
            # Leave already-absolute paths alone
            if os.path.isabs(f):
                continue
            # Make absolute relative to the XML file's directory
            abs_path = (base_dir / f).resolve()
            elem.set("file", abs_path.as_posix())

    # Return as text (no XML declaration; add if you want)
    return ET.tostring(root, encoding="unicode")


def insert_camera(xml_path: Path, camera_xml: str) -> Path:

    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Could not find <worldbody> in the XML.")

    cam_elem = ET.fromstring(camera_xml)

    worldbody.append(cam_elem)
    return ET.tostring(root, encoding="unicode")


    

def render_from_camera(xml_str: str, camera_name: str, width: int, height: int, show: bool = True):
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    if show:
        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(f"Camera: {camera_name}")
        plt.tight_layout()
        plt.show()

    return rgb


def main():
    p = argparse.ArgumentParser(description="Insert a camera into a MuJoCo XML and render from it.")
    p.add_argument("--xml", required=True, help="Path to the original MuJoCo XML.")
    p.add_argument("--camera_name", default="left_camera", help="Camera name to render from.")
    p.add_argument("--save", default=None, help="Optional output image path (e.g., out.png).")
    args = p.parse_args()

    xml_path = Path(args.xml)
    cam_xml = DEFAULT_CAMERA_SNIPPET


    cam_elem = ET.fromstring(cam_xml)
    cam_elem.attrib["name"] = args.camera_name
    width, height = parse_resolution_from_camera(cam_elem)
        
    cam_xml = ET.tostring(cam_elem, encoding="unicode")


    xml_str = insert_camera(xml_path, cam_xml)
    xml_str = expand_asset_paths(xml_path, xml_str)
    rgb = render_from_camera(xml_str, args.camera_name, width, height, show=True)

    if args.save:
        plt.imsave(args.save, rgb)
        print(f"[saved] {args.save}")


if __name__ == "__main__":
    main()
