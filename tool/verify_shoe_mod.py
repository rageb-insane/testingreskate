"""Checks a shoe mod built by make_shoe_mod.py the way ReSkate and the game will read it:
every file and hash, every reference between assets, the texture headers, and the mesh
itself (decoded back out of the mod's own chunks and rendered to <mod>-check.png).

    python verify_shoe_mod.py <mod folder> [--game E:\\...\\Skate]
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402
import meshset  # noqa: E402
import tangentspace  # noqa: E402
from make_deck_mod import EBXTOOL, DEFAULT_GAME, BUNDLE_REF_TABLE, TEXTURE_RES, djb  # noqa: E402
from make_shoe_mod import ITEM_COLLECTION, MESH_BUNDLE_REF, bundle_ref, ebx_file_guid, bundles_of_other_mods  # noqa: E402
import shoe_mesh  # noqa: E402
from shoe_mesh import MESHSET_RES  # noqa: E402

problems = []


def check(ok, message):
    print(('  ok   ' if ok else '  FAIL ') + message)
    if not ok:
        problems.append(message)


def dump(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ebx') as f:
        f.write(data)
    try:
        r = subprocess.run([EBXTOOL, 'dump', f.name], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        return r.stdout
    finally:
        os.unlink(f.name)


def ebx_file(text): return re.search(r'^file (\S+)', text, re.M).group(1)
def ebx_root(text): return re.search(r'^instance #0 guid=(\S+)', text, re.M).group(1)
def ebx_imports(text): return re.findall(r'^import #\d+ file=(\S+) inst=(\S+)', text, re.M)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mod')
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('--as-modelled', action='store_true',
                   help='built with --fit model: the legwear check is reported, not held against it')
    a = p.parse_args()
    g = fb.GameData(a.game)
    flags, bundles, chunks = fb.read_toc(open(os.path.join(a.mod, 'Win32', 'items.toc'), 'rb').read())
    check(flags == 3 and len(bundles) == 3, f'items.toc: {len(bundles)} bundles, {len(chunks)} chunks')
    toc = {c.guid: c for c in chunks}
    _, base_bundles, _ = g.toc('Win32/items.toc')
    base = {b.name: b for b in base_bundles}
    cas_size = os.path.getsize(os.path.join(a.mod, 'Win32', 'configurations', 'layout', 'initialinstallpackage', 'cas_01.cas'))
    check(all(c.offset + c.size <= cas_size for c in chunks), 'TOC chunks lie inside cas_01.cas')

    found = {}          # (kind, name) -> (asset, data)
    where = {}          # (kind, name) -> bundle name
    metas = {}          # bundle name -> {h64: chunk metadata entry}
    ebx_files = {}      # bundle name -> file guids of every EBX it holds
    for b in bundles:
        print(b.name.split('/')[-1])
        files, _ = fb.read_bundle_region(b.region)
        ebx, res, ch, meta = g.manifest(files, a.mod)
        bf, _ = fb.read_bundle_region(base[b.name].region)
        bebx, bres, bch, _ = g.manifest(bf)
        check(len(files) == 1 + len(ebx) + len(res) + len(ch), 'region files match the manifest')
        check([x.name for x in ebx[:len(bebx)]] == [x.name for x in bebx] and
              [x.name for x in res[:len(bres)]] == [x.name for x in bres] and
              [x.guid for x in ch[:len(bch)]] == [x.guid for x in bch], 'base assets kept, additions appended')
        tree, _ = fb.read_db(meta, 0)
        check(len(tree['children']) == len(ch), 'chunk metadata has one entry per chunk')
        # the game finds a chunk's entry by its h64 (the base lists them in another order)
        metas[b.name] = {struct.unpack('<Q', fb.db_field(e, 'h64')['value'])[0]: e for e in tree['children']}
        ebx_files[b.name] = {ebx_file_guid(g.payload(files[1 + i], a.mod)) for i in range(len(ebx))}
        for i, x in enumerate(ebx + res + ch):
            where[(x.kind, x.name)] = b.name
            f = files[1 + i]
            if not f.patch:
                continue
            raw = g.read(f, a.mod)
            check(f.offset + f.size <= cas_size and fb.sha1(raw) == x.sha1, f'stored bytes + sha1: {x.kind} {x.name[-56:]}')
            data = fb.decode_cas(raw, a.game)
            check(len(data) == (x.logical_size if x.kind == 'chunk' else x.original_size), '  size matches')
            if x.kind == 'res':
                check(x.res_id & 1 == 1, f'  resource id {x.res_id:#x} has bit 0 set')
            found[(x.kind, x.name)] = (x, data)

    def ebxs(prefix):
        return [(n, v) for (k, n), v in found.items() if k == 'ebx' and n.startswith(prefix)]

    def toc_data(guid):
        c = toc[guid]
        return fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size), a.mod), a.game)

    def check_texture(name, header):
        mips, first = header[30], header[31]
        sizes = struct.unpack_from(f'<{mips}I', header, 56)
        total = struct.unpack_from('<I', header, 116)[0]
        guid = header[40:56]
        check(struct.unpack_from('<Q', header, 120)[0] == djb(name, 64), f'texture name hash: {name[-48:]}')
        check(guid in toc and sum(sizes) == total, '  texture chunk in items.toc, mip sizes add up')
        if guid not in toc:
            return
        c = toc[guid]
        raw = g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size), a.mod)
        at, logical, starts = 0, 0, {}
        while at < len(raw):
            starts[logical] = at
            comp = struct.unpack_from('<H', raw, at + 4)[0]
            logical += fb.be32(raw, at) & 0xFFFFFF
            at += 8 + struct.unpack_from('>H', raw, at + 6)[0] + (((comp >> 8) & 0x0F) << 16)
        check(logical == total, f'  streamed chunk is {logical} bytes')
        mip_start = [sum(sizes[:k]) for k in range(mips)]
        check(all(m in starts for m in mip_start[:max(first, 2) + 1]), '  mips 0..2 and every separately streamed mip start a compressed block')
        if mips > 1:
            check(first in (0, header[32]), f"  first mip {first}: " + ('the bundle carries the whole texture' if first == 0
                                                                       else "the game's own for this texture group"))
        want = (starts.get(mip_start[1], -1) if mips > 1 else 0, starts.get(mip_start[2], -1) if mips > 2 else 0)
        check(struct.unpack_from('<II', header, 0) == want, f'  header stream offsets {want}')
        bundled = [(n, v) for (k, n), v in found.items() if k == 'chunk' and v[0].guid == guid]
        if mips > 1:    # thumbnails (one mip) are streamed whole, like the game's own
            check(len(bundled) == 1, '  whole texture in the bundle' if first == 0 else '  mip tail copy in the bundle')
        if bundled:
            n, (x, data) = bundled[0]
            check(x.logical_offset == mip_start[first] and data == toc_data(guid)[mip_start[first]:],
                  '  bundle copy starts at the first mip')
            entry = metas[where[('chunk', n)]].get(djb(name, 64))
            m = fb.db_field(entry, 'meta') if entry else None
            fm = fb.db_field(m, 'firstMip') if m else None
            check(fm is not None and struct.unpack('<i', fm['value'])[0] == first,
                  f'  chunk metadata: h64 and firstMip {first}')

    print('item')
    items = ebxs('items/cust_shoes/')
    check(len(items) == 1, 'one new shoe item')
    item = dump(items[0][1][1])
    key = re.search(r'Key = "([^"]+)"', item).group(1)
    check(re.search(r'HashedAssetKey = (\d+)', item).group(1) == str(djb(key)), f'HashedAssetKey of {key}')
    paths = re.findall(r'AssetTypeId = (\d+)\s+AssetName = "([^"]+)"', item)
    by_type = {int(t): n for t, n in paths}
    item_imports = ebx_imports(item)
    coll = dump(found[('ebx', ITEM_COLLECTION)][1])
    check(f'file={ebx_file(item)} inst={ebx_root(item)}' in coll, 'season item collection lists the item')

    print('thumbnails')
    for name, (asset, data) in ebxs('thumbnail/cdn/'):
        t = dump(data)
        check((ebx_file(t), ebx_root(t)) in item_imports, f'item imports {name.split("/")[-1]}')
        check(re.search(r'NameHash = (\d+)', t).group(1) == str(djb(name)), '  NameHash')
        tool_file, tool_inst = ebx_imports(t)[0]
        tool = [(n, d) for n, (x, d) in ebxs('thumbnail/tool/') if ebx_file(dump(d)) == tool_file]
        check(len(tool) == 1, '  points at its texture')
        rid = int(re.search(r'res\(0x([0-9a-f]+)\)', dump(tool[0][1])).group(1), 16)
        r = found[('res', tool[0][0])]
        check(r[0].res_id == rid, '  texture resource id')
        check_texture(tool[0][0], r[1])

    print('mesh')
    table_res = found[('res', BUNDLE_REF_TABLE)]
    table = bundleref.Table(table_res[1], table_res[0].res_meta)
    geo = [(n, v) for n, v in ebxs('characters/maincharacters/reskate/') if n.endswith('/' + by_type.get(2, '?'))]
    check(len(geo) == 1, f'geometry {by_type.get(2)} is in the mesh bundle')
    check(geo[0][0] in table.presets and table.bundle_name_token(table.presets[geo[0][0]])[0] == MESH_BUNDLE_REF,
          '  bundle-reference table loads it from the sneaker mesh bundle')
    geo_text = dump(geo[0][1][1])
    mesh = [(n, v) for n, v in ebxs('characters/maincharacters/reskate/') if n.endswith('_mesh')]
    mesh_text = dump(mesh[0][1][1])
    check((ebx_file(mesh_text), ebx_root(mesh_text)) in ebx_imports(geo_text), '  geometry -> mesh')
    check(re.search(r'NameHash = (\d+)', mesh_text).group(1) == str(djb(mesh[0][0])), '  mesh NameHash')
    var = [(n, v) for n, v in ebxs('characters/maincharacters/reskate/') if '_variation' in n]
    var_text = dump(var[0][1][1])
    mesh_insts = set(re.findall(r'^instance #\d+ guid=(\S+)', mesh_text, re.M))
    check(all(i in mesh_insts for f, i in ebx_imports(var_text) if f == ebx_file(mesh_text)) and
          any(f == ebx_file(mesh_text) for f, _ in ebx_imports(var_text)), '  variation -> mesh and its material')
    rid = int(re.search(r'MeshSetResource = res\(0x([0-9a-f]+)\)', mesh_text).group(1), 16)
    ms_asset, ms_data = found[('res', mesh[0][0])]
    check(ms_asset.res_id == rid and ms_asset.res_type == MESHSET_RES, '  mesh -> MeshSet resource')
    ms = meshset.MeshSet(ms_data)
    check(ms.name == mesh[0][0] and ms.name_hash == djb(mesh[0][0]), '  MeshSet name and hash')
    meta = ms_asset.res_meta
    check(struct.unpack_from('<I', meta, 0)[0] + struct.unpack_from('<I', meta, 4)[0] <= len(ms_data), '  MeshSet metadata sizes')
    render = None
    for i, lod in enumerate(ms.lods):
        check(lod.chunk in toc, f'LOD{i}: chunk in items.toc ({lod.index_size + lod.vertex_size} bytes)')
        data = toc_data(lod.chunk)
        check(len(data) >= lod.vertex_size + lod.index_size, '  chunk holds vertex + index buffers')
        check(struct.unpack_from('<I', ms_data, lod.at + 0xA4)[0] == djb(lod.name[5:]), '  LOD name hash')
        bundled = [(n, v) for (k, n), v in found.items() if k == 'chunk' and v[0].guid == lod.chunk]
        check(bool(bundled) == (i > 0), f'  {"in" if i else "not in"} the mesh bundle, like the game\'s own LODs')
        if bundled:
            n, (x, copy) = bundled[0]
            check(copy == data and djb(ms.name, 64) in metas[where[('chunk', n)]], '  bundle copy and metadata')
        for s in lod.sections:
            v = meshset.decode_section(lod, s, data)
            tri = v['triangles']
            w = v['bone_w0'].astype(int).sum(1) + (v['bone_w1'].astype(int).sum(1) if 'bone_w1' in v else 0)
            bones = np.concatenate([v['bone_idx0']] + ([v['bone_idx1']] if 'bone_idx1' in v else []), 1)
            lo, hi = np.array(s.bbox[0]), np.array(s.bbox[1])
            pos = v['pos'][:, :3]
            ok = (tri.max() < s.vertex_count and (w == 255).all() and bones.max() < 386 and
                  (pos >= lo - 1e-3).all() and (pos <= hi + 1e-3).all())
            check(ok, f'  {s.material}: {s.vertex_count} verts, {s.prims} tris, weights 255, in bounds')
            if 'tangent_frame' in v:
                n, t, b = tangentspace.decode(v['tangent_frame'])
                p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
                fn = np.cross(p1 - p0, p2 - p0)
                agree = ((fn * (n[tri[:, 0]] + n[tri[:, 1]] + n[tri[:, 2]])).sum(1) > 0).mean()
                check(agree > 0.95, f'  normals face out of the triangles ({agree:.1%})')
                if i == 0:
                    render = (pos, tri, n)
    if render:
        reference = shoe_mesh.Reference(g)
        legwear = shoe_mesh.lower_leg_garments(g, reference.positions[:, 1].max() + 0.02)
        showing, total = shoe_mesh.legwear_showing(render[0], reference.positions, legwear)
        if render[0][:, 1].max() >= shoe_mesh.HIGH_TOP:
            # a high-top is shaped like the game's Dunk High: hold it to what the game ships for that shoe
            dunk = shoe_mesh.Reference(g, shoe_mesh.DUNK_HIGH_BUNDLE, None).positions
            d_show, d_total = shoe_mesh.legwear_showing(dunk, reference.positions, legwear)
            allowed = max(0.05, 1.5 * d_show / d_total)
            baseline = f"the game's Dunk High: {d_show} of {d_total}"
        else:
            allowed, baseline = 0.05, "the game's sneaker: 0"
        if a.as_modelled:
            print(f'  note legwear: {showing} of {total} places would show through ({baseline}); the shoe is '
                  'as modelled (--fit model), not shaped for the pants')
        else:
            check(total and showing <= allowed * total,
                  f'legwear stays inside the collar: {showing} of {total} places would show through '
                  f'({baseline}; allowed: {allowed:.0%})')
        from PIL import Image, ImageDraw
        pos, tri, n = render
        size = 900
        img = Image.new('RGB', (size * 2, size), (28, 28, 34))
        d = ImageDraw.Draw(img)
        light = np.array([0.4, 0.8, 0.45])
        light /= np.linalg.norm(light)
        # left: the outer side of the x>0 shoe; right: both shoes from above
        for panel, (ax, ay, view, keep) in enumerate(((2, 1, np.array([1.0, 0, 0]), pos[:, 0] > 0),
                                                       (0, 2, np.array([0, 1.0, 0]), pos[:, 0] == pos[:, 0]))):
            t = tri[keep[tri].all(1)]
            p0, p1, p2 = pos[t[:, 0]], pos[t[:, 1]], pos[t[:, 2]]
            fn = np.cross(p1 - p0, p2 - p0)
            front = fn @ view > 0
            t = t[front]
            q = pos[:, [ax, ay]]
            mn, mx = q[keep].min(0), q[keep].max(0)
            sc = (size - 60) / (mx - mn).max()
            pts = (q - mn) * sc + 30
            if panel == 0:
                pts[:, 1] = size - pts[:, 1]
            pts[:, 0] += panel * size
            shade = np.clip(n[t].mean(1) @ light, 0, 1)
            for k in np.argsort((pos[t] @ view).mean(1)):
                c = int(60 + 170 * shade[k])
                d.polygon([tuple(pts[j]) for j in t[k]], fill=(c, c, c))
        out = os.path.normpath(a.mod) + '-check.png'
        img.save(out)
        print('  rendered the decoded LOD0 mesh to', out)

    print('appearance')
    preset = [(n, v) for n, v in ebxs('characters/customization/reskate/textureedits/') if n.endswith('/' + by_type.get(3, '?'))]
    check(len(preset) == 1, f'preset {by_type.get(3)} is in the colorway bundle')
    colorway = where[('ebx', preset[0][0])]
    check(preset[0][0] in table.presets and table.bundle_name_token(table.presets[preset[0][0]])[0] == bundle_ref(colorway),
          f'  bundle-reference table loads it from {bundle_ref(colorway).split("/")[-1]}')
    others = bundles_of_other_mods(a.game, os.path.basename(os.path.normpath(a.mod))).get(colorway, [])
    check(not others, '  no other installed mod ships that bundle (ReSkate would merge it and drop the '
                      'textures\' firstMip metadata)' + (f': {", ".join(others)}' if others else ''))
    pre_imports = ebx_imports(dump(preset[0][1][1]))
    for n, (x, d) in ebxs('characters/customization/reskate/textureedits/'):
        if n.endswith('_ap'):
            missing = [f for f, _ in ebx_imports(dump(d)) if f not in ebx_files[colorway]]
            check(not missing, f'  every import of preset {n.split("/")[-1][:8]} is in the same bundle' +
                  (f' (missing {missing})' if missing else ''))
    tinted = re.findall(r'boxed\(18 at=\d+ bytes=([0-9a-f]{24})', dump(preset[0][1][1]))
    check(tinted and all(struct.unpack('<3f', bytes.fromhex(t)) == struct.unpack('<3f', bytes.fromhex(tinted[0]))
                         for t in tinted), f'  {len(tinted)} colour tints, all the same')
    textures = [(n, v) for n, v in ebxs('characters/customization/reskate/textureedits/') if '_ap_' in n]
    for name, (asset, data) in textures:
        t = dump(data)
        rid = int(re.search(r'res\(0x([0-9a-f]+)\)', t).group(1), 16)
        r = found[('res', name)]
        check(r[0].res_id == rid and r[0].res_type == TEXTURE_RES, f'texture {name.split("_ap_")[1][:8]} -> resource')
        check_texture(name, r[1])
    linked = [n for n, (x, d) in textures if (ebx_file(dump(d)), ebx_root(dump(d))) in pre_imports]
    check(len(linked) == 5, f'item preset imports all five textures ({len(linked)})')

    print('\nALL CHECKS PASSED' if not problems else f'\n{len(problems)} PROBLEM(S)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
