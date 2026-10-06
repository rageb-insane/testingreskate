"""Minimal Wavefront OBJ reader with the same output as gltf.load: triangle meshes (polygons
fanned into triangles), one primitive per object/material, uv flipped to glTF's top-left origin."""
import os

import numpy as np

from gltf import Primitive


def load(path):
    """Returns (primitives, doc), shaped like gltf.load's. doc['materials'] lists the materials
    glTF-style ({'name', 'pbrMetallicRoughness': {'baseColorFactor'}}; the .mtl's Kd, taken as
    linear like a glTF factor, or a light grey without a .mtl), and each primitive's material is
    an index into it, so a model painted by material colours alone goes through
    gltf.colour_palette like a textureless .glb (--colours, --gold and --unmirror pick its parts
    by the .mtl's material names). doc['obj_textures'] maps material index -> base colour texture
    path (map_Kd), found beside the model when the .mtl names a path from another machine (None
    when it is nowhere)."""
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

    # the .mtl files: diffuse colour and colour texture per material name
    colours, maps = {}, {}
    for lib in mtllibs:
        p = os.path.join(folder, lib)
        if not os.path.isfile(p):
            continue
        current = None
        for line in open(p, encoding='utf-8', errors='replace'):
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'newmtl':
                current = ' '.join(parts[1:])
            elif parts[0] == 'Kd' and current is not None and len(parts) >= 4:
                colours[current] = [min(max(float(x), 0.0), 1.0) for x in parts[1:4]]
            elif parts[0] == 'map_Kd' and current is not None:
                maps[current] = parts[-1]
    materials, index_of, textures = [], {}, {}

    def material_index(name):
        if name not in index_of:
            index_of[name] = len(materials)
            materials.append({'name': name or '', 'pbrMetallicRoughness': {
                'baseColorFactor': colours.get(name, [0.8, 0.8, 0.8]) + [1.0]}})
            if name in maps:
                textures[index_of[name]] = _find(folder, maps[name])
        return index_of[name]

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
                               indices=np.array(tris, np.int64), material=material_index(material)))
    return prims, {'obj_textures': textures, 'materials': materials, '_folder': folder}


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
