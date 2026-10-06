"""Checks a deck mod built by make_deck_mod.py: re-reads every file the way ReSkate
does and confirms the pieces point at each other.

    python verify_mod.py <mod folder> [--game E:\\...\\Skate] [--with-mod <other mod folder>]
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402
from make_deck_mod import (EBXTOOL, SHARED_BUNDLE, DECK_BUNDLE, ITEM_COLLECTION, BUNDLE_REF_TABLE, DEFAULT_GAME,
                           TEXTURE_RES, djb)  # noqa: E402

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


def field(text, name):
    m = re.search(rf'^\s*{name} = (.*)$', text, re.M)
    return m.group(1).strip() if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mod')
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('--with-mod', help='another installed mod to test merging against')
    a = p.parse_args()
    g = fb.GameData(a.game)
    flags, bundles, chunks = fb.read_toc(open(os.path.join(a.mod, 'Win32', 'items.toc'), 'rb').read())
    check(flags == 3 and len(bundles) == 2, f'items.toc holds {len(bundles)} bundles, {len(chunks)} chunks')
    toc_chunks = {c.guid: c for c in chunks}
    _, base_bundles, _ = g.toc('Win32/items.toc')
    base = {b.name: b for b in base_bundles}
    cas_path = os.path.join(a.mod, 'Win32', 'configurations', 'layout', 'initialinstallpackage', 'cas_01.cas')
    cas_size = os.path.getsize(cas_path)

    found = {}
    for b in bundles:
        print(b.name)
        files, inline = fb.read_bundle_region(b.region)
        ebx, res, ch, meta = g.manifest(files, a.mod)
        bf, _ = fb.read_bundle_region(base[b.name].region)
        bebx, bres, bch, bmeta = g.manifest(bf)
        check(len(files) == 1 + len(ebx) + len(res) + len(ch), 'region files match the manifest')
        check(all(f.offset + f.size <= cas_size for f in files if f.patch), 'patched files lie inside cas_01.cas')
        check([x.name for x in ebx[:len(bebx)]] == [x.name for x in bebx] and
              [x.name for x in res[:len(bres)]] == [x.name for x in bres] and
              [x.guid for x in ch[:len(bch)]] == [x.guid for x in bch], 'base asset lists kept, additions appended')
        def sections(fl, e, r):
            return fl[1:1 + e], fl[1 + e:1 + e + r], fl[1 + e + r:]
        base_files_same = all((f.loc(), f.offset, f.size) == (o.loc(), o.offset, o.size)
                              for mine, theirs in zip(sections(files, len(ebx), len(res)), sections(bf, len(bebx), len(bres)))
                              for f, o in zip(mine, theirs) if not f.patch)
        check(base_files_same, 'unpatched entries still point at the base game')
        tree, _ = fb.read_db(meta, 0)
        check(len(tree['children']) == len(ch), 'chunk metadata has one entry per chunk')
        for x, entry in zip(ch, tree['children']):
            first = fb.db_field(fb.db_field(entry, 'meta') or {'children': []}, 'firstMip')
            x.first_mip = struct.unpack('<i', first['value'])[0] if first else None
            x.h64 = struct.unpack('<Q', fb.db_field(entry, 'h64')['value'])[0]
        for i, x in enumerate(ebx + res + ch):
            f = files[1 + i]
            if not f.patch:
                continue
            raw = g.read(f, a.mod)
            check(fb.sha1(raw) == x.sha1, f'sha1 matches stored bytes: {x.kind} {x.name[-60:]}')
            data = fb.decode_cas(raw, a.game)
            if x.kind == 'chunk':
                check(len(data) == x.logical_size, 'chunk size matches')
            else:
                check(len(data) == x.original_size, 'original size matches')
            found[(x.kind, x.name)] = (x, data)
            if x.kind == 'res':
                check(x.res_id & 1 == 1, f'resource id {x.res_id:#x} has bit 0 set: {x.name[-60:]}')

    def check_texture(name, header):
        """The resource header against the chunk data the game will stream and load."""
        mips, first = header[30], header[31]
        sizes = struct.unpack_from(f'<{mips}I', header, 56)
        total = struct.unpack_from('<I', header, 116)[0]
        guid = header[40:56]
        check(struct.unpack_from('<Q', header, 120)[0] == djb(name, 64), f'header name hash: {name[-50:]}')
        check(sum(sizes) == total, 'mip sizes add up to the chunk size')
        toc = toc_chunks.get(guid)
        check(toc is not None, 'texture chunk is listed in items.toc')
        if toc is None:
            return
        raw = g.read(fb.FileInfo(toc.patch, toc.install_chunk, toc.archive, toc.offset, toc.size), a.mod)
        full = fb.decode_cas(raw, a.game)
        check(len(full) == total, f'streamed chunk decodes to {len(full)} bytes')
        at, logical, starts = 0, 0, {}
        while at < len(raw):
            starts[logical] = at
            dsize = fb.be32(raw, at) & 0xFFFFFF
            comp = struct.unpack_from('<H', raw, at + 4)[0]
            at += 8 + struct.unpack_from('>H', raw, at + 6)[0] + (((comp >> 8) & 0x0F) << 16)
            logical += dsize
        mip_start = [sum(sizes[:k]) for k in range(mips)]
        check(all(mip_start[k] in starts for k in range(min(first, mips - 1) + 1)),
              'every separately loaded mip starts a compressed block')
        want = (starts.get(mip_start[1], -1) if mips > 1 else 0, starts.get(mip_start[2], -1) if mips > 2 else 0)
        check(struct.unpack_from('<II', header, 0) == want, f'header stream offsets {want}')
        bundle_chunk = [v for (k, n), v in found.items() if k == 'chunk' and v[0].guid == guid]
        if first or bundle_chunk:
            check(len(bundle_chunk) == 1, 'bundle carries the mip tail')
            x, data = bundle_chunk[0]
            check(x.logical_offset == mip_start[first] and x.logical_size == total - mip_start[first] and
                  data == full[mip_start[first]:], f'bundle tail is mips {first}+ of the streamed chunk')
            check(x.first_mip == first and x.h64 == djb(name, 64), 'chunk metadata firstMip and h64')

    print('cross references')
    item = [v for (k, n), v in found.items() if k == 'ebx' and n.startswith('items/board_bottomart/')]
    check(len(item) == 1, 'one new deck item')
    item_text = dump(item[0][1])
    key = field(item_text, 'Key').strip('"')
    check(field(item_text, 'HashedAssetKey').startswith(str(djb(key))), f'HashedAssetKey is the hash of {key}')
    item_file = re.search(r'^file (\S+)', item_text, re.M).group(1)
    item_inst = re.search(r'^instance #0 guid=(\S+)', item_text, re.M).group(1)
    preset_leaf = field(item_text, 'AssetName').strip('"')
    imports = dict((m[0], m[1]) for m in re.findall(r'^import #\d+ file=(\S+) inst=(\S+)', item_text, re.M))

    # thumbnails: item -> cdn -> tool texture -> resource -> chunk
    for suffix in ('', '_lrg'):
        cdn = [v for (k, n), v in found.items() if k == 'ebx' and n.startswith('thumbnail/cdn/') and n.endswith(
            ('_lrg' if suffix else '')) and (suffix or not n.endswith('_lrg'))][0]
        cdn_text = dump(cdn[1])
        cdn_file = re.search(r'^file (\S+)', cdn_text, re.M).group(1)
        check(imports.get(cdn_file) == re.search(r'^instance #0 guid=(\S+)', cdn_text, re.M).group(1),
              f'item imports its thumbnail{suffix}')
        check(field(cdn_text, 'NameHash').startswith(str(djb(cdn[0].name))), 'thumbnail NameHash')
        tool_file, tool_inst = re.search(r'^import #0 file=(\S+) inst=(\S+)', cdn_text, re.M).groups()
        tool = [v for (k, n), v in found.items() if k == 'ebx' and n.startswith('thumbnail/tool/') and
                (n.endswith('_lrg') == bool(suffix))][0]
        tool_text = dump(tool[1])
        check(re.search(r'^file (\S+)', tool_text, re.M).group(1) == tool_file and
              re.search(r'^instance #0 guid=(\S+)', tool_text, re.M).group(1) == tool_inst, 'thumbnail record -> texture')
        res_id = int(re.search(r'res\(0x([0-9a-f]+)\)', tool_text).group(1), 16)
        tres = [v for (k, n), v in found.items() if k == 'res' and n == tool[0].name][0]
        check(tres[0].res_id == res_id and tres[0].res_type == TEXTURE_RES, 'texture -> resource id')
        check_texture(tres[0].name, tres[1])

    # deck: preset -> texture asset -> resource -> chunk; bundle-reference row
    preset = [v for (k, n), v in found.items() if k == 'ebx' and n.endswith('/' + preset_leaf)]
    check(len(preset) == 1, f'item preset {preset_leaf} is in the deck bundle')
    preset_text = dump(preset[0][1])
    tex = [v for (k, n), v in found.items() if k == 'ebx' and n.startswith(preset[0][0].name + '_')][0]
    tex_text = dump(tex[1])
    tex_file = re.search(r'^file (\S+)', tex_text, re.M).group(1)
    tex_inst = re.search(r'^instance #0 guid=(\S+)', tex_text, re.M).group(1)
    check(f'file={tex_file} inst={tex_inst}' in preset_text, 'preset imports the deck texture')
    res_id = int(re.search(r'res\(0x([0-9a-f]+)\)', tex_text).group(1), 16)
    tres = [v for (k, n), v in found.items() if k == 'res' and n == tex[0].name][0]
    check(tres[0].res_id == res_id, 'deck texture -> resource id')
    width, height = struct.unpack_from('<HH', tres[1], 22)
    check((width, height, tres[1][31]) == (512, 2048, 3), f'deck texture {width}x{height}, first mip {tres[1][31]}')
    check_texture(tres[0].name, tres[1])

    table_res = found[('res', BUNDLE_REF_TABLE)]
    table = bundleref.Table(table_res[1], table_res[0].res_meta)
    check(preset[0][0].name in table.presets, 'bundle-reference table names the preset')
    bundle_path = table.bundle_name_token(table.presets[preset[0][0].name])[0]
    check('win32/' + bundle_path + '_cas_main_bundlereftable' == DECK_BUNDLE, f'preset loads from {bundle_path}')

    coll = dump(found[('ebx', ITEM_COLLECTION)][1])
    check(f'file={item_file} inst={item_inst}' in coll, 'season item collection lists the item')

    # ReSkate-style merge with another mod editing the same assets
    if a.with_mod:
        print('merge with', os.path.basename(a.with_mod))
        _, ob, _ = fb.read_toc(open(os.path.join(a.with_mod, 'Win32', 'items.toc'), 'rb').read())
        other = {b.name: b for b in ob}
        sb = base[SHARED_BUNDLE]
        bf, _ = fb.read_bundle_region(sb.region)
        bebx, bres, _, _ = g.manifest(bf)
        of, _ = fb.read_bundle_region(other[SHARED_BUNDLE].region)
        oebx, ores, _, _ = g.manifest(of, a.with_mod)
        with tempfile.TemporaryDirectory() as t:
            def save(name, data):
                path = os.path.join(t, name)
                open(path, 'wb').write(data)
                return path
            ib = next(i for i, x in enumerate(bebx) if x.name == ITEM_COLLECTION)
            io = next(i for i, x in enumerate(oebx) if x.name == ITEM_COLLECTION)
            r = subprocess.run([EBXTOOL, 'merge', save('base', g.payload(bf[1 + ib])), os.path.join(t, 'merged'),
                                save('other', g.payload(of[1 + io], a.with_mod)), save('mine', found[('ebx', ITEM_COLLECTION)][1])],
                               capture_output=True, text=True)
            check(r.returncode == 0, 'item collections merge: ' + (r.stdout.strip() or r.stderr.strip()))
            merged = dump(open(os.path.join(t, 'merged'), 'rb').read())
            check(f'file={item_file} inst={item_inst}' in merged, 'merged collection still lists the new deck')
        rb_ = next(i for i, x in enumerate(bres) if x.name == BUNDLE_REF_TABLE)
        ro = next(i for i, x in enumerate(ores) if x.name == BUNDLE_REF_TABLE)
        base_table = bundleref.Table(g.payload(bf[1 + len(bebx) + rb_]), bres[rb_].res_meta)
        other_table = bundleref.Table(g.payload(of[1 + len(oebx) + ro], a.with_mod), ores[ro].res_meta)
        same_bundles = table.data[table.bundles:table.lookups] == base_table.data[base_table.bundles:base_table.lookups]
        kept = all(table.presets.get(p) == b for p, b in base_table.presets.items())
        added = set(table.presets) - set(base_table.presets)
        check(same_bundles and kept and added == {preset[0][0].name}, 'bundle-reference edit is additions-only (mergeable)')
        check(not (added & (set(other_table.presets) - set(base_table.presets))), 'no preset clashes with the other mod')

    print('\nALL CHECKS PASSED' if not problems else f'\n{len(problems)} PROBLEM(S)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
