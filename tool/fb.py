"""Minimal Frostbite (skate.) TOC / bundle / CAS reader-writer, ported from
ReSkate's Engine/Resource (toc.cpp, binary_bundle.cpp, cas_codec.cpp)."""
import struct, ctypes, os, hashlib

TOC_ENVELOPE = 0x22C
FNV_PRIME = 0x01000193
SALT = 0x7065636E
BUNDLE_MAGIC = 0xED1CEDB8


def be32(b, o): return struct.unpack_from('>I', b, o)[0]
def bei32(b, o): return struct.unpack_from('>i', b, o)[0]


def toc_hash(key: bytes, seed=0x811C9DC5):
    for byte in key:
        sb = byte - 256 if byte > 127 else byte
        seed = ((seed * FNV_PRIME) & 0xFFFFFFFF) ^ (sb & 0xFFFFFFFF)
    return seed % FNV_PRIME


def guid_str(g: bytes):
    c = bytes([g[3], g[2], g[1], g[0], g[5], g[4], g[7], g[6]]) + g[8:]
    h = c.hex()
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'


# ---------------------------------------------------------------- TOC
class TocBundle:
    def __init__(self, name, region, load_flag=1):
        self.name, self.region, self.load_flag = name, region, load_flag


class TocChunk:
    def __init__(self, guid, patch, install_chunk, archive, offset, size, removed=False):
        self.guid, self.patch, self.install_chunk, self.archive = guid, patch, install_chunk, archive
        self.offset, self.size, self.removed = offset, size, removed


class _Huffman:
    def __init__(self, d, names_off, names_count, table_off, table_count):
        self.words = [be32(d, names_off + 4 * i) for i in range(names_count)]
        self.nodes = []  # (value, left, right)
        by_value = {}
        left = None
        counter = 0
        self.root = None
        for i in range(table_count):
            value = be32(d, table_off + 4 * i)
            node = by_value.get(value)
            if node is None:
                node = len(self.nodes)
                self.nodes.append((value, None, None))
                by_value[value] = node
            if left is None:
                left = node
                continue
            parent = len(self.nodes)
            self.nodes.append((counter, left, node))
            by_value.setdefault(counter, parent)
            self.root = parent
            counter += 1
            left = None

    def decode(self, bit):
        out = []
        nbits = len(self.words) * 32
        while bit < nbits:
            node = self.root
            while self.nodes[node][1] is not None:
                one = (self.words[bit >> 5] >> (bit & 31)) & 1
                node = self.nodes[node][2] if one else self.nodes[node][1]
                bit += 1
            ch = (~self.nodes[node][0]) & 0xFFFF
            if ch == 0:
                return ''.join(out)
            out.append(chr(ch))
        raise ValueError('unterminated Huffman name')


def read_toc(data: bytes):
    assert data[:4] == b'\x00\xd1\xce\x01', 'bad TOC envelope'
    d = data[TOC_ENVELOPE:]
    (bh_off, bd_off, bcount, ch_off, cg_off, ccount, _, _, names_off, cd_off, dcount, flags) = \
        struct.unpack_from('>IIiIIiIIIIiI', d, 0)
    huff = None
    if flags & 4:
        names_count, table_count, table_off = struct.unpack_from('>III', d, 48)
        huff = _Huffman(d, names_off, names_count, table_off, table_count)
    bundles = []
    for i in range(bcount):
        name_off, saf, reg_off = struct.unpack_from('>IIQ', d, bd_off + i * 16)
        size = saf & 0x3FFFFFFF
        if huff:
            name = huff.decode(name_off)
        else:
            name = d[names_off + name_off:d.index(b'\0', names_off + name_off)].decode()
        assert saf >> 30 == 1, 'external bundle data not supported'
        bundles.append(TocBundle(name, d[reg_off:reg_off + size], 1))
    chunks = []
    if ccount:
        words = struct.unpack_from(f'>{dcount}I', d, cd_off)
        for i in range(ccount):
            g = d[cg_off + i * 20: cg_off + i * 20 + 16][::-1]
            enc = bei32(d, cg_off + i * 20 + 16)
            if enc == -1:
                chunks.append(TocChunk(g, False, 0, 0, 0, 0, True))
                continue
            w = enc & 0xFFFFFF
            ident = (words[w] << 32) | words[w + 1]
            chunks.append(TocChunk(g, (ident >> 48) & 0xFF != 0, (ident >> 16) & 0xFFFFFFFF, ident & 0xFFFF,
                                   words[w + 2], words[w + 3]))
    return flags, bundles, chunks


def _perfect_hash(keys):
    n = len(keys)
    hmap = [-1] * n
    order = [None] * n
    if n == 0:
        return hmap, order
    assert len(set(keys)) == n, 'TOC keys must be unique'
    buckets = [[] for _ in range(n)]
    for i, k in enumerate(keys):
        buckets[toc_hash(k) % n].append(i)
    buckets.sort(key=len, reverse=True)  # Python's sort is stable, like std::stable_sort
    used = [False] * n
    bi = 0
    while bi < n and len(buckets[bi]) > 1:
        b = buckets[bi]
        seed = 1
        while True:
            idx = []
            ok = True
            for m in b:
                x = toc_hash(keys[m], seed) % n
                if used[x] or x in idx:
                    ok = False
                    break
                idx.append(x)
            if ok:
                break
            seed += 1
        hmap[toc_hash(keys[b[0]]) % n] = seed
        for j, m in enumerate(b):
            order[idx[j]] = m
            used[idx[j]] = True
        bi += 1
    free = 0
    while bi < n and buckets[bi]:
        while used[free]:
            free = (free + 1) % n
        m = buckets[bi][0]
        hmap[toc_hash(keys[m]) % n] = -free - 1
        order[free] = m
        used[free] = True
        free = (free + 1) % n
        bi += 1
    return hmap, order


def _align(v, a): return (v + a - 1) & ~(a - 1)


def write_patch_toc(bundles, chunks=(), flags=3):
    bundles = list(bundles)
    chunks = list(chunks)
    bmap, border = _perfect_hash([b.name.lower().encode() for b in bundles])
    cmap, corder = _perfect_hash([c.guid for c in chunks])
    names = bytearray()
    name_offs = []
    for si in border:
        name_offs.append(len(names))
        names += bundles[si].name.encode() + b'\0'
    cwords = []
    ctable = bytearray()
    for si in corder:
        c = chunks[si]
        ctable += c.guid[::-1]
        if c.removed:
            ctable += struct.pack('>i', -1)
        else:
            ctable += struct.pack('>I', (0x80 << 24) | len(cwords))
            ident = ((1 if c.patch else 0) << 48) | (c.install_chunk << 16) | c.archive
            cwords += [ident >> 32, ident & 0xFFFFFFFF, c.offset, c.size]
    pos = 48
    bh = pos; pos = _align(pos + len(bundles) * 4, 8)
    bd = pos; pos = _align(pos + len(bundles) * 16, 4)
    chh = pos; pos = _align(pos + len(chunks) * 4, 4)
    cg = pos; pos = _align(pos + len(ctable), 4)
    cd = pos; pos += len(cwords) * 4
    no = pos; pos = _align(pos + len(names), 4)
    reg_offs = []
    for si in border:
        reg_offs.append(pos)
        pos = _align(pos + len(bundles[si].region), 4)
    out = bytearray(TOC_ENVELOPE + pos)
    out[0:4] = b'\x00\xd1\xce\x01'
    struct.pack_into('>IIiIIiIIIIiI', out, TOC_ENVELOPE, bh, bd, len(bundles), chh if chunks else 0,
                     cg if chunks else 0, len(chunks), cd, cd, no, cd, len(cwords), flags)
    B = TOC_ENVELOPE
    if bmap:
        struct.pack_into(f'>{len(bmap)}i', out, B + bh, *bmap)
    for oi, si in enumerate(border):
        b = bundles[si]
        struct.pack_into('>IIQ', out, B + bd + oi * 16, name_offs[oi], (b.load_flag << 30) | len(b.region), reg_offs[oi])
    if chunks:
        struct.pack_into(f'>{len(cmap)}i', out, B + chh, *cmap)
        out[B + cg:B + cg + len(ctable)] = ctable
        struct.pack_into(f'>{len(cwords)}I', out, B + cd, *cwords)
    out[B + no:B + no + len(names)] = names
    for oi, si in enumerate(border):
        r = bundles[si].region
        out[B + reg_offs[oi]:B + reg_offs[oi] + len(r)] = r
    return bytes(out)


# ---------------------------------------------------------------- bundle region
class FileInfo:
    def __init__(self, patch, install_chunk, archive, offset, size):
        self.patch, self.install_chunk, self.archive, self.offset, self.size = patch, install_chunk, archive, offset, size

    def loc(self): return (self.patch, self.install_chunk, self.archive)

    def __repr__(self):
        return f'<{"P" if self.patch else "B"} ic={self.install_chunk} cas_{self.archive:02d} @{self.offset:#x} +{self.size:#x}>'


def read_bundle_region(r: bytes):
    (meta_off, meta_size, loc_off, total, data_off, res_off, chunk_off, _, total2) = struct.unpack_from('>iiIiIIIIi', r, 0)
    assert total == total2 and res_off == data_off == chunk_off, 'bad bundle region'
    inline = r[meta_off:meta_off + meta_size] if meta_size else b''
    files = []
    p = data_off
    cur = None
    for i in range(total):
        flag = r[loc_off + i]
        if flag == 1:
            enc = be32(r, p); p += 4
            cur = ((enc & 0x00FF0000) != 0, (enc >> 8) & 0xFF, enc & 0xFF)
        elif flag != 0:
            ident = struct.unpack_from('>Q', r, p)[0]; p += 8
            cur = (((ident >> 48) & 0xFF) != 0, (ident >> 16) & 0xFFFFFFFF, ident & 0xFFFF)
        off, size = struct.unpack_from('>II', r, p); p += 8
        files.append(FileInfo(*cur, off, size))
    return files, inline


def write_bundle_region(files, inline=b''):
    table = bytearray()
    flags = bytearray()
    prev = None
    for f in files:
        if prev == f.loc():
            flags.append(0)
        else:
            flags.append(0x80)
            table += struct.pack('>Q', ((1 if f.patch else 0) << 48) | (f.install_chunk << 16) | f.archive)
            prev = f.loc()
        table += struct.pack('>II', f.offset, f.size)
    if flags:
        flags[0] |= 0x04
    data_off = 0x24
    meta_off = data_off + len(table) if inline else 0
    loc_off = (meta_off + len(inline)) if inline else data_off + len(table)
    hdr = struct.pack('>iiIiIIIIi', meta_off, len(inline), loc_off, len(files), data_off, data_off, data_off, 0, len(files))
    return hdr + bytes(table) + bytes(inline) + bytes(flags)


# ---------------------------------------------------------------- binary bundle manifest
class Asset:
    def __init__(self, **kw): self.__dict__.update(kw)
    def __repr__(self): return f'<{self.kind} {self.name}>'


def read_binary_bundle(data: bytes):
    declared = be32(data, 0)
    assert declared == len(data) - 4, 'bundle size mismatch'
    magic = be32(data, 4) ^ SALT
    e = '>'
    if magic != BUNDLE_MAGIC:
        magic = struct.unpack_from('<I', data, 4)[0] ^ SALT
        e = '<'
    assert magic == BUNDLE_MAGIC, 'bad bundle magic'
    total, ne, nr, nc, strings, meta_off, meta_size = struct.unpack_from(e + '7I', data, 8)
    strings += 4
    p = 36
    shas = [data[p + i * 20:p + i * 20 + 20] for i in range(total)]
    p += total * 20

    def s(o):
        a = strings + o
        return data[a:data.index(b'\0', a)].decode()
    ebx = []
    for i in range(ne):
        no, osz = struct.unpack_from(e + 'II', data, p); p += 8
        ebx.append(Asset(kind='ebx', name=s(no), sha1=shas[i], original_size=osz))
    res = []
    for i in range(nr):
        no, osz = struct.unpack_from(e + 'II', data, p); p += 8
        res.append(Asset(kind='res', name=s(no), sha1=shas[ne + i], original_size=osz))
    for a in res:
        a.res_type = struct.unpack_from(e + 'I', data, p)[0]; p += 4
    for a in res:
        a.res_meta = data[p:p + 16]; p += 16
    for a in res:
        a.res_id = struct.unpack_from(e + 'Q', data, p)[0]; p += 8
    chunks = []
    for i in range(nc):
        g = data[p:p + 16]
        if e == '>':
            g = g[::-1]
        lo, ls = struct.unpack_from(e + 'II', data, p + 16); p += 24
        chunks.append(Asset(kind='chunk', name=guid_str(g), guid=g, sha1=shas[ne + nr + i],
                            logical_offset=lo, logical_size=ls))
    meta = data[4 + meta_off:4 + meta_off + meta_size] if meta_size else b''
    return ebx, res, chunks, meta


def write_binary_bundle(ebx, res, chunks, meta=b''):
    total = len(ebx) + len(res) + len(chunks)
    strings = bytearray()
    offs = []
    for a in ebx + res:
        offs.append(len(strings))
        strings += a.name.encode() + b'\0'
    sha_start = 36
    ebx_start = sha_start + total * 20
    res_start = ebx_start + len(ebx) * 8
    chunk_start = res_start + len(res) * (8 + 4 + 16 + 8)
    string_start = chunk_start + len(chunks) * 24
    meta_start = string_start + len(strings)
    length = meta_start + len(meta)
    out = bytearray()
    out += struct.pack('>I', length - 4)
    out += struct.pack('<8I', BUNDLE_MAGIC ^ SALT, total, len(ebx), len(res), len(chunks), string_start - 4,
                       (meta_start - 4) if meta else 0, len(meta))
    for a in ebx + res + chunks:
        out += a.sha1
    for i, a in enumerate(ebx):
        out += struct.pack('<II', offs[i], a.original_size)
    for i, a in enumerate(res):
        out += struct.pack('<II', offs[len(ebx) + i], a.original_size)
    for a in res:
        out += struct.pack('<I', a.res_type)
    for a in res:
        out += a.res_meta
    for a in res:
        out += struct.pack('<Q', a.res_id)
    for a in chunks:
        out += a.guid + struct.pack('<II', a.logical_offset, a.logical_size)
    out += strings + meta
    assert len(out) == length
    return bytes(out)


# ---------------------------------------------------------------- CAS blocks
_oodle = None


def oodle(game_root):
    global _oodle
    if _oodle is None:
        _oodle = ctypes.WinDLL(os.path.join(game_root, 'oo2core_9_win64.dll'))
        _oodle.OodleLZ_Decompress.restype = ctypes.c_int64
        _oodle.OodleLZ_Decompress.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int]
        _oodle.OodleLZ_Compress.restype = ctypes.c_int64
        _oodle.OodleLZ_Compress.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        _oodle.OodleLZ_GetCompressedBufferSizeNeeded.restype = ctypes.c_int64
        _oodle.OodleLZ_GetCompressedBufferSizeNeeded.argtypes = [ctypes.c_int, ctypes.c_int64]
    return _oodle


def decode_cas(enc: bytes, game_root):
    out = bytearray()
    p = 0
    while p < len(enc):
        dsize = be32(enc, p)
        comp = struct.unpack_from('<H', enc, p + 4)[0]
        esize = struct.unpack_from('>H', enc, p + 6)[0]
        flags = comp >> 8
        esize += (flags & 0x0F) << 16
        assert (dsize & 0xFF000000) == 0, 'dictionary blocks unsupported'
        comp &= 0x7F
        p += 8
        block = enc[p:p + esize]
        p += esize
        if comp == 0:
            out += block
        elif comp in (0x11, 0x15, 0x19):
            o = oodle(game_root)
            buf = ctypes.create_string_buffer(dsize)
            n = o.OodleLZ_Decompress(block, len(block), buf, dsize, 1, 0, 0, None, 0, None, None, None, 0, 3)
            assert n == dsize, 'oodle decode failed'
            out += buf.raw
        else:
            raise NotImplementedError(f'CAS compression {comp:#x}')
    return bytes(out)


def encode_cas(data: bytes, game_root=None, compress=True, block_size=0x10000):
    out = bytearray()
    for i in range(max(1, (len(data) + block_size - 1) // block_size)):
        inp = data[i * block_size:(i + 1) * block_size]
        payload, comp = inp, 0
        if compress and inp:
            o = oodle(game_root)
            cap = o.OodleLZ_GetCompressedBufferSizeNeeded(8, len(inp))
            buf = ctypes.create_string_buffer(cap)
            n = o.OodleLZ_Compress(8, inp, len(inp), buf, 4, None, None, None, None, 0)
            if 0 < n < len(inp):
                payload, comp = buf.raw[:n], 0x11
        packed = ((len(inp) & 0xFFFFFF) << 32) | (comp << 24) | (0x7 << 20) | len(payload)
        out += struct.pack('>Q', packed) + payload
    return bytes(out)


def sha1(b): return hashlib.sha1(b).digest()


# ---------------------------------------------------------------- native DB (layout.toc)
def read_db(data: bytes, pos=0):
    """Returns (node, end). node = dict(type, name, children | value bytes)."""
    def var():
        nonlocal pos
        v = 0; shift = 0
        while True:
            b = data[pos]; pos += 1
            v |= (b & 127) << shift
            if not b & 128: return v
            shift += 7

    def node():
        nonlocal pos
        tag = data[pos]; pos += 1
        t = tag & 31
        if not t:
            return None
        name = None
        if not tag & 128:
            end = data.index(b'\0', pos)
            name = data[pos:end].decode()
            pos = end + 1
        if t in (1, 2):
            size = var()
            end = pos + size
            children = []
            terminated = False
            while pos < end:
                c = node()
                if c is None:
                    terminated = True
                    break
                children.append(c)
            assert pos == end, 'DB container length mismatch'
            return dict(type=t, name=name, children=children, terminated=terminated)
        if t in (7, 19):
            n = var()
            v = data[pos:pos + n]; pos += n
            return dict(type=t, name=name, value=v)
        size = {6: 1, 8: 4, 11: 4, 9: 8, 12: 8, 15: 16, 16: 20}[t]
        v = data[pos:pos + size]; pos += size
        return dict(type=t, name=name, value=v)
    n = node()
    return n, pos


def write_db(node) -> bytes:
    def var(v):
        out = bytearray()
        while True:
            b = v & 127
            v >>= 7
            if not v:
                out.append(b)
                return bytes(out)
            out.append(b | 128)

    out = bytearray([node['type'] | (0 if node['name'] is not None else 128)])
    if node['name'] is not None:
        out += node['name'].encode() + b'\0'
    if node['type'] in (1, 2):
        body = b''.join(write_db(c) for c in node['children']) + (b'\0' if node.get('terminated') else b'')
        out += var(len(body)) + body
    elif node['type'] in (7, 19):
        out += var(len(node['value'])) + node['value']
    else:
        out += node['value']
    return bytes(out)


def db_field(node, key):
    for c in node.get('children', []):
        if c['name'] == key:
            return c
    return None


class GameData:
    """Reads the base game's superbundle TOCs, bundle manifests and payloads."""

    def __init__(self, game_root):
        self.root = game_root
        self.data = os.path.join(game_root, 'Data')
        layout = open(os.path.join(self.data, 'layout.toc'), 'rb').read()
        tree, _ = read_db(layout, TOC_ENVELOPE)
        self.layout = tree
        self.chunk_dirs = {}
        man = db_field(tree, 'installManifest')
        for entry in (db_field(man, 'installChunks') or {}).get('children', []):
            name = db_field(entry, 'name')
            idx = db_field(entry, 'persistentIndex')
            if name and idx and idx['type'] == 8:
                self.chunk_dirs[struct.unpack('<I', idx['value'])[0]] = name['value'].rstrip(b'\0').decode().lower()
        self._cas = {}

    def cas_path(self, root, install_chunk, archive):
        return os.path.join(root, 'Win32', self.chunk_dirs[install_chunk], f'cas_{archive:02d}.cas')

    def read(self, f, patch_root=None):
        root = patch_root if f.patch else self.data
        path = self.cas_path(root, f.install_chunk, f.archive)
        with open(path, 'rb') as h:
            h.seek(f.offset)
            return h.read(f.size)

    def payload(self, f, patch_root=None):
        raw = self.read(f, patch_root)
        return decode_cas(raw, self.root)

    def manifest(self, files, patch_root=None):
        raw = self.read(files[0], patch_root)
        try:
            return read_binary_bundle(raw)
        except Exception:
            return read_binary_bundle(decode_cas(raw, self.root))

    def toc(self, rel):
        return read_toc(open(os.path.join(self.data, rel), 'rb').read())
