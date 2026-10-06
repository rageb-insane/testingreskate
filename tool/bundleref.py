"""Character bundle-reference tables (characters/customization/configs/cas_main_bundlereftable):
which bundle the game loads for each appearance preset. Port of ReSkate's
Engine/Resource/bundle_ref_table.cpp (reader + insert)."""
import struct

NONE = 0xFFFFFFFF


def fnv64(name: str):
    v = 0xCBF29CE484222325
    for c in name.encode():
        if 65 <= c <= 90:
            c += 32
        v = ((v * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF) ^ c
    return v


def leaf(path): return path.rsplit('/', 1)[-1]


class Table:
    def __init__(self, data: bytes, meta: bytes):
        self.data, self.meta = bytearray(data), bytearray(meta)
        self._parse()

    def _parse(self):
        d = self.data
        self.payload = len(d) - 20
        assert struct.unpack_from('<I', self.meta, 0)[0] == self.payload and struct.unpack_from('<I', self.meta, 4)[0] == 20
        for i in range(5):
            assert struct.unpack_from('<I', d, self.payload + i * 4)[0] == i * 8
        self.pool = struct.unpack_from('<Q', d, 16)[0]
        self.bundles = struct.unpack_from('<Q', d, 24)[0]
        self.lookups = struct.unpack_from('<Q', d, 8)[0]
        assert self.pool == 128 and struct.unpack_from('<I', d, 92)[0] == 1
        count = struct.unpack_from('<I', d, 72)[0]
        self.bundle_count = struct.unpack_from('<I', d, 76)[0]
        assert self.bundles + self.bundle_count * 8 == self.lookups and self.lookups + count * 16 == self.payload
        self.rows = [struct.unpack_from('<QII', d, self.lookups + i * 16) for i in range(count)]
        self._cache = {}
        self.presets = {}
        for h, b, tok in self.rows:
            self.presets[self.text(tok)] = b

    def text(self, token, depth=0):
        if token == NONE:
            return ''
        if token in self._cache:
            return self._cache[token]
        d = self.data
        at = self.pool + (token & 0x7FFFFF)
        length = token >> 24
        frag = at + 4 if token & 0x800000 else self.pool + struct.unpack_from('<I', d, at + 4)[0]
        parent = struct.unpack_from('<I', d, at)[0]
        s = self.text(parent, depth + 1) + bytes(d[frag:frag + length]).decode('ascii')
        self._cache[token] = s
        return s

    def bundle_name_token(self, index):
        tok, flags = struct.unpack_from('<II', self.data, self.bundles + index * 8)
        return self.text(tok), flags

    def bundle_index(self, bundle_path):
        for i in range(1, self.bundle_count):
            if self.bundle_name_token(i)[0].lower() == bundle_path.lower():
                return i
        raise KeyError(bundle_path)

    def insert(self, path: str, bundle: int):
        """Adds full-path and leaf rows for `path`, pointing at bundle index `bundle`."""
        full, short = fnv64(path), fnv64(leaf(path))
        held = {r[0] for r in self.rows}
        if full in held:
            raise ValueError(path + ' collides with an existing lookup')
        with_leaf = full != short and short not in held
        growth = (4 + len(path) + 15) & ~15
        offset = self.bundles - self.pool
        token = offset | 0x800000 | (len(path) << 24)
        bundles, lookups = self.bundles + growth, self.lookups + growth
        payload = self.payload + growth + (32 if with_leaf else 16)
        d = self.data
        out = bytearray(payload + 20)
        out[:self.bundles] = d[:self.bundles]
        struct.pack_into('<I', out, self.bundles, NONE)
        out[self.bundles + 4:self.bundles + 4 + len(path)] = path.encode()
        out[bundles:bundles + (self.lookups - self.bundles)] = d[self.bundles:self.lookups]
        rows = list(self.rows) + [(full, bundle, token)] + ([(short, bundle, token)] if with_leaf else [])
        rows.sort(key=lambda r: r[0])
        for i, (h, b, t) in enumerate(rows):
            struct.pack_into('<QII', out, lookups + i * 16, h, b, t)
        out[payload:] = d[self.payload:]
        struct.pack_into('<Q', out, 8, lookups)
        struct.pack_into('<Q', out, 24, bundles)
        struct.pack_into('<I', out, 72, len(rows))
        struct.pack_into('<I', self.meta, 0, payload)
        self.data = out
        self._parse()
        return with_leaf
