"""Checks a pack built by make_modpack.py against the mods it was made from: every stored
file and hash, every bundle's manifest and chunk metadata, and that everything each mod adds
(items, presets, textures, meshes, bundle-reference rows, item-list entries) is in the pack.

    python verify_modpack.py <pack folder> <mod folder name>... [--game E:\\...\\Skate]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402
from make_deck_mod import EBXTOOL, DEFAULT_GAME, BUNDLE_REF_TABLE  # noqa: E402

problems = []


def check(ok, message):
    print(('  ok   ' if ok else '  FAIL ') + message)
    if not ok:
        problems.append(message)


def imports(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ebx') as f:
        f.write(data)
    try:
        out = subprocess.run([EBXTOOL, 'dump', f.name], capture_output=True, text=True).stdout
    finally:
        os.unlink(f.name)
    return set(re.findall(r'^import #\d+ file=(\S+) inst=(\S+)', out, re.M))


def bundle_contents(g, root, b):
    files, inline = fb.read_bundle_region(b.region)
    ebx, res, chunks, meta = fb.read_binary_bundle(inline) if inline else g.manifest(files, root)
    first = 0 if inline else 1
    return files, first, ebx, res, chunks, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pack')
    p.add_argument('mods', nargs='+')
    p.add_argument('--game', default=DEFAULT_GAME)
    a = p.parse_args()
    g = fb.GameData(a.game)
    flags, bundles, chunks = fb.read_toc(open(os.path.join(a.pack, 'Win32', 'items.toc'), 'rb').read())
    check(flags == 3, f'items.toc: {len(bundles)} bundles, {len(chunks)} chunks')
    by_name = {b.name: b for b in bundles}
    toc_guids = {c.guid for c in chunks}
    _, base_bundles, _ = g.toc('Win32/items.toc')
    base = {b.name: b for b in base_bundles}

    def archive(f):
        return os.path.join(a.pack, 'Win32', *g.chunk_dirs[f.install_chunk].split('/'), f'cas_{f.archive:02d}.cas')

    sizes = {}
    print('stored files')
    bad_bounds = bad_sha = 0
    pack = {}
    for b in bundles:
        files, first, ebx, res, ch, meta = bundle_contents(g, a.pack, b)
        if len(files) != first + len(ebx) + len(res) + len(ch):
            check(False, f'{b.name}: region files do not match its manifest')
        if meta:
            tree, _ = fb.read_db(meta, 0)
            if len(tree['children']) != len(ch):
                check(False, f'{b.name.split("/")[-1]}: {len(ch)} chunks, {len(tree["children"])} metadata entries')
        for x, f in zip(ebx + res + ch, files[first:]):
            if not f.patch:
                continue
            path = archive(f)
            sizes.setdefault(path, os.path.getsize(path) if os.path.isfile(path) else -1)
            if f.offset + f.size > sizes[path]:
                bad_bounds += 1
                continue
            with open(path, 'rb') as h:
                h.seek(f.offset)
                raw = h.read(f.size)
            if fb.sha1(raw) != x.sha1:
                bad_sha += 1
        pack[b.name] = (files, first, ebx, res, ch)
    for c in chunks:
        if c.patch and not c.removed:
            path = os.path.join(a.pack, 'Win32', *g.chunk_dirs[c.install_chunk].split('/'), f'cas_{c.archive:02d}.cas')
            sizes.setdefault(path, os.path.getsize(path) if os.path.isfile(path) else -1)
            if c.offset + c.size > sizes[path]:
                bad_bounds += 1
    check(bad_bounds == 0, f'every file and chunk lies inside its archive ({len(sizes)} archives)')
    check(bad_sha == 0, 'every stored file matches its manifest sha1' + (f' ({bad_sha} do not)' if bad_sha else ''))

    pack_table = None
    sb = [n for n in pack if n.endswith('cas_main_sharedbundle')]
    if sb:
        files, first, ebx, res, ch = pack[sb[0]]
        for x, f in zip(res, files[first + len(ebx):]):
            if x.name == BUNDLE_REF_TABLE:
                pack_table = bundleref.Table(fb.decode_cas(open(archive(f), 'rb').read()[f.offset:f.offset + f.size], g.root)
                                             if f.patch else g.payload(f), x.res_meta)

    def pack_payload(f):
        if not f.patch:
            return g.payload(f)
        with open(archive(f), 'rb') as h:
            h.seek(f.offset)
            return fb.decode_cas(h.read(f.size), g.root)

    for name in a.mods:
        root = os.path.join(a.game, 'Mods', name)
        print(name)
        _, mb, mc = fb.read_toc(open(os.path.join(root, 'Win32', 'items.toc'), 'rb').read())
        check(all(c.guid in toc_guids for c in mc), f'  its {len(mc)} items.toc chunks')
        missing, items_lost, rows_lost = [], 0, 0
        for b in mb:
            files, first, ebx, res, ch, meta = bundle_contents(g, root, b)
            if b.name not in pack:
                missing.append(b.name)
                continue
            pfiles, pfirst, pebx, pres, pch = pack[b.name]
            have = {(x.kind, x.name) for x in pebx + pres} | {x.guid for x in pch}
            want = {(x.kind, x.name) for x in ebx + res} | {x.guid for x in ch}
            missing += [str(k) for k in want - have]
            if b.name not in base:
                continue
            bf, _ = fb.read_bundle_region(base[b.name].region)
            be, br, _, _ = g.manifest(bf)
            bsha = {(x.kind, x.name): x for x in be + br}
            bfile = {(x.kind, x.name): f for x, f in zip(be + br, bf[1:])}
            pmap = {(x.kind, x.name): f for x, f in zip(pebx + pres, pfiles[pfirst:])}
            for x, f in zip(ebx, files[first:]):
                k = (x.kind, x.name)
                if k in bsha and bsha[k].sha1 != x.sha1 and x.name.endswith('_itemcollection') and k in pmap:
                    mine = imports(g.payload(f, root)) - imports(g.payload(bfile[k]))
                    items_lost += len(mine - imports(pack_payload(pmap[k])))
            for x, f in zip(res, files[first + len(ebx):]):
                if x.name == BUNDLE_REF_TABLE and pack_table is not None:
                    t = bundleref.Table(g.payload(f, root), x.res_meta)
                    held = {r[0] for r in pack_table.rows}
                    rows_lost += sum(1 for h, _, tok in t.rows if h not in held)
        check(not missing, f'  every bundle and asset it ships ({len(mb)} bundles)' + (f': missing {missing[:3]}' if missing else ''))
        check(items_lost == 0, '  its item-list entries' + (f': {items_lost} missing' if items_lost else ''))
        check(rows_lost == 0, '  its bundle-reference rows' + (f': {rows_lost} missing' if rows_lost else ''))

    print('\nALL CHECKS PASSED' if not problems else f'\n{len(problems)} PROBLEM(S)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
