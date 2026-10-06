"""skate. (Frostbite) MeshSet resources: reader for skinned character meshes.

Layout learned from the game's own sneaker mesh and a working custom-mesh mod.
Offsets inside the resource are stored relative to byte 0x10 (`BASE`).
"""
import struct

BASE = 0x10
LOD_SIZE = 0xC0
SECTION_SIZE = 0x180

USAGE = {1: 'pos', 2: 'bone_idx0', 3: 'bone_idx1', 4: 'bone_w0', 5: 'bone_w1',
         0x21: 'uv0', 0x22: 'uv1', 0x34: 'tangent_frame'}
FORMAT_SIZE = {2: 8, 3: 12, 6: 4, 8: 8, 0x0d: 4, 0x17: 8}


def u32(b, o): return struct.unpack_from('<I', b, o)[0]
def u64(b, o): return struct.unpack_from('<Q', b, o)[0]


def cstr(b, o):
    return b[o:b.index(b'\0', o)].decode()


class Section:
    pass


class Lod:
    pass


class MeshSet:
    def __init__(self, data):
        self.data = data
        d = data
        self.bbox = (struct.unpack_from('<3f', d, 0x10), struct.unpack_from('<3f', d, 0x20))
        lod_offsets = [u64(d, 0x30 + 8 * i) for i in range(7)]
        self.name = cstr(d, u64(d, 0x68) + BASE)
        self.full_name = cstr(d, u64(d, 0x70) + BASE)
        self.name_hash = u32(d, 0x78)
        self.lods = []
        for off in lod_offsets:
            if not off:
                continue
            self.lods.append(self._lod(off + BASE))

    def _lod(self, at):
        d = self.data
        lod = Lod()
        lod.at = at
        lod.section_count = u32(d, at + 0x08)
        sections_at = u32(d, at + 0x0C) + BASE
        lod.index_size, lod.vertex_size = struct.unpack_from('<II', d, at + 0x58)
        lod.chunk = d[at + 0x74:at + 0x84]
        lod.name = cstr(d, u32(d, at + 0x8C) + BASE)
        lod.sections = [self._section(sections_at + i * SECTION_SIZE) for i in range(lod.section_count)]
        return lod

    def _section(self, at):
        d = self.data
        s = Section()
        s.at = at
        s.material = cstr(d, u64(d, at + 0x08) + BASE)
        s.prims, s.start_index, s.vertex_offset, s.vertex_count = struct.unpack_from('<4I', d, at + 0x20)
        s.elements = []
        for i in range(16):
            usage, fmt, offset, stream = struct.unpack_from('<4B', d, at + 0x70 + 4 * i)
            if usage == 0:  # unused slots read 00 00 ff 00
                continue
            s.elements.append((usage, fmt, offset, stream))
        s.strides = [struct.unpack_from('<BB', d, at + 0xB0 + 2 * i)[0] for i in range(8)]
        s.element_count, s.stream_count = struct.unpack_from('<BB', d, at + 0xD0)
        s.strides = s.strides[:s.stream_count]
        s.stride = sum(s.strides)
        s.bbox = (struct.unpack_from('<3f', d, at + 0x150), struct.unpack_from('<3f', d, at + 0x160))
        return s


def decode_section(lod, section, chunk):
    """Vertex attributes and triangles of one section from its LOD chunk (vertex
    buffer stream-major per section, then the u16 index buffer)."""
    import numpy as np
    n = section.vertex_count
    out = {}
    at = section.vertex_offset
    for stream, stride in enumerate(section.strides):
        for usage, fmt, _, st in section.elements:
            if st != stream:
                continue
            name = USAGE.get(usage, hex(usage))
            if fmt == 3:      # Float3 (the game's pants)
                out[name] = np.frombuffer(chunk, np.float32, n * 3, at).reshape(-1, 3).astype(np.float64)
            elif fmt == 2:    # Float2
                out[name] = np.frombuffer(chunk, np.float32, n * 2, at).reshape(-1, 2).astype(np.float64)
            elif fmt == 8:
                out[name] = np.frombuffer(chunk, np.float16, n * 4, at).reshape(-1, 4).astype(np.float64)
            elif fmt == 6:
                out[name] = np.frombuffer(chunk, np.float16, n * 2, at).reshape(-1, 2).astype(np.float64)
            elif fmt == 0x17:
                out[name] = np.frombuffer(chunk, np.uint16, n * 4, at).reshape(-1, 4)
            elif fmt == 0x0D:
                out[name] = np.frombuffer(chunk, np.uint8, n * 4, at).reshape(-1, 4)
        at += stride * n
    out['triangles'] = np.frombuffer(chunk, np.uint16, section.prims * 3,
                                     lod.vertex_size + section.start_index * 2).reshape(-1, 3).astype(np.int64)
    return out
