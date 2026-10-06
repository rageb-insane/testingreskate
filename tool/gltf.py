"""Minimal glTF 2.0 reader (.gltf with its files, or a single .glb): triangle meshes with world
transforms applied, and the images they use."""
import base64
import io
import json
import os
import struct

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def _matrix(node):
    if 'matrix' in node:
        return np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T   # column-major
    m = np.eye(4)
    if 'scale' in node:
        m = np.diag(list(node['scale']) + [1.0]) @ m
    if 'rotation' in node:
        x, y, z, w = node['rotation']
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        rm = np.eye(4)
        rm[:3, :3] = r
        m = rm @ m
    if 'translation' in node:
        t = np.eye(4)
        t[:3, 3] = node['translation']
        m = t @ m
    return m


class Primitive:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def load(path):
    """Returns (primitives, gltf json). Each primitive has world-space positions,
    normals, uv (or None), triangle indices, material index and node name."""
    folder = os.path.dirname(path)
    with open(path, 'rb') as f:
        data = f.read()
    binary = None
    if data[:4] == b'glTF':                  # .glb: a JSON chunk, then the binary buffer
        at, doc = 12, None
        while at + 8 <= len(data):
            size, kind = struct.unpack_from('<I4s', data, at)
            if kind == b'JSON':
                doc = json.loads(data[at + 8:at + 8 + size].decode('utf-8'))
            elif kind == b'BIN' + bytes(1):
                binary = data[at + 8:at + 8 + size]
            at += 8 + size
        if doc is None:
            raise SystemExit(f'{os.path.basename(path)}: a .glb without its JSON part')
    else:
        doc = json.loads(data.decode('utf-8-sig'))
    buffers = []
    for b in doc.get('buffers', []):
        uri = b.get('uri')
        if uri is None:
            buffers.append(binary)
        elif uri.startswith('data:'):
            buffers.append(base64.b64decode(uri.split(',', 1)[1]))
        else:
            buffers.append(open(os.path.join(folder, uri), 'rb').read())
    doc['_folder'], doc['_buffers'] = folder, buffers

    def accessor(i):
        a = doc['accessors'][i]
        view = doc['bufferViews'][a['bufferView']]
        data = buffers[view['buffer']]
        dtype = np.dtype(COMPONENT[a['componentType']])
        width = WIDTH[a['type']]
        start = view.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = view.get('byteStride', 0) or dtype.itemsize * width
        raw = np.frombuffer(data, dtype=np.uint8, count=stride * (a['count'] - 1) + dtype.itemsize * width, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(a['count'], dtype.itemsize * width), strides=(stride, 1))
        out = np.ascontiguousarray(rows).view(dtype).reshape(a['count'], width)
        if a.get('normalized'):
            out = out.astype(np.float64) / np.iinfo(dtype).max
        return out

    prims = []

    def walk(index, parent):
        node = doc['nodes'][index]
        world = parent @ _matrix(node)
        if 'mesh' in node:
            for p in doc['meshes'][node['mesh']]['primitives']:
                if p.get('mode', 4) != 4:
                    continue
                pos = accessor(p['attributes']['POSITION']).astype(np.float64)
                pos = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3]
                normal = None
                if 'NORMAL' in p['attributes']:
                    nm = np.linalg.inv(world[:3, :3]).T
                    normal = accessor(p['attributes']['NORMAL']).astype(np.float64) @ nm.T
                    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
                uv = accessor(p['attributes']['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in p['attributes'] else None
                idx = accessor(p['indices']).reshape(-1, 3).astype(np.int64) if 'indices' in p else \
                    np.arange(len(pos)).reshape(-1, 3)
                if np.linalg.det(world[:3, :3]) < 0:      # mirrored node: keep triangles front-facing
                    idx = idx[:, ::-1]
                prims.append(Primitive(name=node.get('name', ''), node=index, positions=pos, normals=normal, uv=uv,
                                       indices=idx, material=p.get('material')))
        for child in node.get('children', []):
            walk(child, world)

    scene = doc['scenes'][doc.get('scene', 0)]
    for root in scene['nodes']:
        walk(root, np.eye(4))
    return prims, doc


def image(doc, index):
    """Image `index` of a document from load(), as a PIL image: a file beside the model, a data
    URI, or (in a .glb) bytes in a buffer view."""
    from PIL import Image
    im = doc['images'][index]
    if 'bufferView' in im:
        view = doc['bufferViews'][im['bufferView']]
        start = view.get('byteOffset', 0)
        raw = doc['_buffers'][view['buffer']][start:start + view['byteLength']]
    elif im.get('uri', '').startswith('data:'):
        raw = base64.b64decode(im['uri'].split(',', 1)[1])
    else:
        from urllib.parse import unquote
        raw = open(os.path.join(doc['_folder'], unquote(im['uri'])), 'rb').read()
    out = Image.open(io.BytesIO(raw))
    out.load()
    return out


def base_colour_image(doc, material):
    """The base colour texture of a material, or None."""
    if material is None:
        return None
    pbr = doc['materials'][material].get('pbrMetallicRoughness', {})
    tex = pbr.get('baseColorTexture')
    if tex is None:
        return None
    return image(doc, doc['textures'][tex['index']]['source'])


def colour_palette(prims, doc, tile=8, colours=None):
    """For a model painted by material colours alone (no textures): a small texture with one flat
    square per material colour (baseColorFactor, linear -> sRGB), every primitive's UVs pointed at
    its material's square and every primitive given material 0. colours: a colourway, [(part of a
    material name, '#rrggbb')] tried in order ('*' matches any). Returns (palette image,
    {material name: uv of its square})."""
    from PIL import Image
    mats = doc.get('materials', [])

    def override(m):
        name = (mats[m].get('name', '') if m is not None and m < len(mats) else '').lower()
        for key, hexc in colours or ():
            if key == '*' or key.lower() in name:
                return np.array([int(hexc.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)]) / 255
        return None
    keys = sorted({p.material for p in prims}, key=lambda m: (m is None, m))
    n = int(np.ceil(np.sqrt(max(len(keys), 1))))
    size = 1
    while size < n * tile:
        size *= 2
    img = np.zeros((size, size, 3), np.uint8)
    where = {}
    for k, m in enumerate(keys):
        c = (mats[m].get('pbrMetallicRoughness', {}).get('baseColorFactor', [1, 1, 1, 1]) if m is not None and m < len(mats)
             else [0.8, 0.8, 0.8, 1])
        srgb = np.clip(np.where(np.array(c[:3]) <= 0.0031308, np.array(c[:3]) * 12.92,
                                1.055 * np.power(np.clip(c[:3], 0, 1), 1 / 2.4) - 0.055), 0, 1)
        if override(m) is not None:
            srgb = override(m)
        y, x = divmod(k, n)
        img[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile] = np.round(srgb * 255)
        where[m] = ((x + 0.5) * tile / size, (y + 0.5) * tile / size)
    names = {(mats[m].get('name', '') if m is not None and m < len(mats) else ''): uv for m, uv in where.items()}
    for p in prims:
        p.uv = np.tile(np.array(where[p.material], float), (len(p.positions), 1))
        p.material = 0
    return Image.fromarray(img, 'RGB'), names
