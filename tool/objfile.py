"""Minimal Wavefront OBJ reader with the same output as gltf.load: triangle meshes (polygons
fanned into triangles), one primitive per object/material, uv flipped to glTF's top-left origin."""
import os

import numpy as np

from gltf import Primitive


def load(path):
    """Returns (primitives, doc). doc['obj_textures'] maps material name -> base colour texture
    path (map_Kd), found beside the model when the .mtl names a path from another machine."""
    folder = os.path.dirname(os.path.abspath(path))
    v, vt, vn = [], [], []
    groups = {}                        # (object, material) -> list of faces [(vi, ti, ni), ...]
    obj, mat, mtllibs = '', None, []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == 'v':
                v.append([float(x) for x in parts[1:4]])
            elif tag == 'vt':
                vt.append([float(x) for x in parts[1:3]])
            elif tag == 'vn':
                vn.append([float(x) for x in parts[1:4]])
            elif tag == 'f':
                corners = []
                for c in parts[1:]:
                    idx = (c.split('/') + ['', ''])[:3]
                    corners.append(tuple(int(x) if x else 0 for x in idx))
                groups.setdefault((obj, mat), []).append(corners)
            elif tag in ('o', 'g'):
                obj = ' '.join(parts[1:])
            elif tag == 'usemtl':
                mat = ' '.join(parts[1:])
            elif tag == 'mtllib':
                mtllibs.append(' '.join(parts[1:]))
    v, vt, vn = np.array(v, np.float64), np.array(vt or [[0, 0]], np.float64), np.array(vn or [[0, 0, 0]], np.float64)

    def fix(i, n):                     # 1-based, negative = from the end
        return i - 1 if i > 0 else n + i
    prims = []
    for (name, material), faces in groups.items():
        keys, index, tris = {}, [], []
        for corners in faces:
            ids = []
            for vi, ti, ni in corners:
                key = (fix(vi, len(v)), fix(ti, len(vt)) if ti else -1, fix(ni, len(vn)) if ni else -1)
                if key not in keys:
                    keys[key] = len(index)
                    index.append(key)
                ids.append(keys[key])
            for k in range(1, len(ids) - 1):
                tris.append((ids[0], ids[k], ids[k + 1]))
        index = np.array(index)
        uv = vt[np.maximum(index[:, 1], 0)].copy()
        uv[:, 1] = 1 - uv[:, 1]                       # OBJ's origin is bottom-left
        normals = vn[np.maximum(index[:, 2], 0)] if (index[:, 2] >= 0).all() else None
        if normals is not None:
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        prims.append(Primitive(name=name, node=None, positions=v[index[:, 0]], normals=normals,
                               uv=uv if (index[:, 1] >= 0).all() else None,
                               indices=np.array(tris, np.int64), material=material))
    textures = {}
    for lib in mtllibs:
        p = os.path.join(folder, lib)
        if not os.path.isfile(p):
            continue
        current = None
        for line in open(p, encoding='utf-8', errors='replace'):
            parts = line.split()
            if parts and parts[0] == 'newmtl':
                current = ' '.join(parts[1:])
            elif parts and parts[0] == 'map_Kd' and current is not None:
                textures[current] = _find(folder, parts[-1])
    return prims, {'obj_textures': textures, '_folder': folder}


def _find(folder, named):
    """A texture path from the .mtl, or the file of that name anywhere under the model's folder."""
    if os.path.isfile(named):
        return named
    if os.path.isfile(os.path.join(folder, named)):
        return os.path.join(folder, named)
    base = os.path.basename(named.replace('\\', '/'))
    for dirpath, _, files in os.walk(folder):
        if base in files:
            return os.path.join(dirpath, base)
    return None
