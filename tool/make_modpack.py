"""Merges several installed cosmetic mods (shoes, clothing, decks, ...: mods that only patch
Win32/items.toc) into one ReSkate mod: one folder, one entry in the launcher, one zip.

    python make_modpack.py --name "My Pack" Mod-One Mod-Two ...   [--install]
    python make_modpack.py --interactive                          (pick from the installed mods)

How the mods are combined (the same rules ReSkate applies at launch, minus its known issue):
  * every mod's cas archives are copied in under new numbers, and every reference to them
    (bundle files, items.toc chunks) is rewritten to match;
  * a bundle only one of the mods ships passes through as it is;
  * a bundle several mods ship gets one manifest: the game's assets plus everything each mod
    adds or changes. An EBX asset two mods changed differently (the season item lists every
    cosmetic mod adds to) is merged with ReSkate's own EBX merge; the character bundle-
    reference table gets every mod's rows; anything else changed two ways keeps the copy of
    the mod listed first (highest priority, as in mods.json). A change to an EBX asset also
    reaches the pack's other copies of that asset, in other bundles;
  * chunk metadata (each texture's firstMip) is kept from every mod. ReSkate's own merge keeps
    only the first copy's, which makes the game crash or show noise on textures that rely on it.
Map mods (levels, globals.toc, ...) are refused: ReSkate merges those with map-specific rules.
"""
import argparse
import collections
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402
from make_deck_mod import EBXTOOL, DEFAULT_GAME, BUNDLE_REF_TABLE, PATCH_DIRECTORY  # noqa: E402

PACKAGE_FILES = {'manifest.json', 'icon.png', 'README.md', 'CHANGELOG.md', 'initfs_win32', 'layout.toc',
                 '.reskate-studio-patch'}


def read(path):
    with open(path, 'rb') as f:
        return f.read()


def is_skate_running():
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Skate.exe'], capture_output=True, text=True).stdout
    return 'skate.exe' in out.lower()


def is_launcher_running():
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq ReSkateLauncher.exe'], capture_output=True, text=True).stdout
    return 'reskatelauncher.exe' in out.lower()


# ------------------------------------------------------------------ the mods
class Mod:
    def __init__(self, g, root):
        self.root, self.name = root, os.path.basename(root)
        win32 = os.path.join(root, 'Win32')
        extra = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                rel = os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, '/')
                if rel in PACKAGE_FILES or rel == 'Win32/items.toc' or rel.endswith('.cas'):
                    continue
                extra.append(rel)
        tocs = [f for f in extra if f.endswith('.toc')]
        if not os.path.isfile(os.path.join(win32, 'items.toc')) or tocs or any(f.startswith('reskate-') for f in extra):
            raise SystemExit(f'{self.name} is not a cosmetic mod (it patches more than Win32/items.toc: '
                             f'{", ".join((tocs or extra)[:3]) or "no items.toc"}). Maps and other mods that '
                             'change levels need ReSkate\'s own map merge; keep them installed separately.')
        self.extra = [f for f in extra if not f.lower().endswith(('.png', '.md', '.txt', '.json'))]
        self.flags, self.bundles, self.chunks = fb.read_toc(read(os.path.join(win32, 'items.toc')))
        manifest = os.path.join(root, 'manifest.json')
        self.manifest = json.load(open(manifest, encoding='utf-8-sig')) if os.path.isfile(manifest) else {}
        readme = os.path.join(root, 'README.md')
        self.readme = open(readme, encoding='utf-8', errors='replace').read() if os.path.isfile(readme) else ''
        icon = os.path.join(root, 'icon.png')
        self.icon = icon if os.path.isfile(icon) else None


# ------------------------------------------------------------------ archives
class Archives:
    """Gives every mod's patch archives new numbers inside the pack, and holds the pack's own."""

    def __init__(self, g, mods):
        self.g, self.map, self.sources = g, {}, []
        used = {}
        for mod in mods:
            locs = set()
            for b in mod.bundles:
                files, _ = fb.read_bundle_region(b.region)
                locs |= {f.loc() for f in files if f.patch}
            locs |= {(True, c.install_chunk, c.archive) for c in mod.chunks if c.patch and not c.removed}
            for patch, ic, arch in sorted(locs):
                n = used.get(ic, 0) + 1
                used[ic] = n
                self.map[(mod.name, ic, arch)] = n
                src = os.path.join(mod.root, 'Win32', *g.chunk_dirs[ic].split('/'), f'cas_{arch:02d}.cas')
                if not os.path.isfile(src):
                    raise SystemExit(f'{mod.name}: {src} is missing')
                self.sources.append((src, ic, n))
        self.own_chunk = next(k for k, v in g.chunk_dirs.items() if v == PATCH_DIRECTORY)
        self.own_archive = used.get(self.own_chunk, 0) + 1
        self.own = bytearray()

    def remap(self, mod, f):
        if not f.patch:
            return f
        return fb.FileInfo(True, f.install_chunk, self.map[(mod.name, f.install_chunk, f.archive)], f.offset, f.size)

    def add(self, payload):
        """Stores a new payload in the pack's own archive; returns (file info, sha1 of the stored bytes)."""
        enc = fb.encode_cas(payload, self.g.root, True)
        f = fb.FileInfo(True, self.own_chunk, self.own_archive, len(self.own), len(enc))
        self.own += enc
        return f, hashlib.sha1(enc).digest()

    def write(self, out_root):
        for src, ic, n in self.sources:
            dst = os.path.join(out_root, 'Win32', *self.g.chunk_dirs[ic].split('/'), f'cas_{n:02d}.cas')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        if self.own:
            dst = os.path.join(out_root, 'Win32', *self.g.chunk_dirs[self.own_chunk].split('/'), f'cas_{self.own_archive:02d}.cas')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, 'wb') as f:
                f.write(self.own)


# ------------------------------------------------------------------ merging
def ebx_merge(base, edits, work):
    paths = []
    for i, data in enumerate([base] + edits):
        p = os.path.join(work, f'm{i}.ebx')
        with open(p, 'wb') as f:
            f.write(data)
        paths.append(p)
    out = os.path.join(work, 'merged.ebx')
    r = subprocess.run([EBXTOOL, 'merge', paths[0], out, *paths[1:]], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return read(out)


def table_merge(base_data, base_meta, copies):
    """The character bundle-reference table with every copy's added rows."""
    table = bundleref.Table(base_data, base_meta)
    held = {r[0] for r in table.rows}
    added, skipped = 0, []
    for data, meta in copies:
        t = bundleref.Table(data, meta)
        for h, b, tok in t.rows:
            path = t.text(tok)
            if h in held or h != bundleref.fnv64(path):        # known, or a leaf row (insert adds those)
                continue
            try:
                table.insert(path, table.bundle_index(t.bundle_name_token(b)[0]))
            except (KeyError, ValueError) as e:
                skipped.append(f'{path}: {e}')
                continue
            held = {r[0] for r in table.rows}
            added += 1
    return bytes(table.data), bytes(table.meta), added, skipped


def build_pack(g, mods, name, author, out_root, log=print):
    """Writes the merged mod to out_root/<author>-<name>; returns its folder."""
    work = tempfile.mkdtemp(prefix='modpack_')
    try:
        return _build(g, mods, name, author, out_root, work, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _build(g, mods, name, author, out_root, work, log):
    package = ''.join(c if c.isalnum() else '_' for c in name).strip('_') or 'Modpack'
    folder = f'{author}-{package}'
    mod_dir = os.path.join(out_root, folder)
    arch = Archives(g, mods)
    _, base_bundles, _ = g.toc('Win32/items.toc')
    base = {b.name: b for b in base_bundles}
    notes = []

    # every bundle copy: (mod, name, files, ebx, res, chunks, meta tree, inline manifest)
    copies = {}
    for mod in reversed(mods):                     # lowest priority first, as ReSkate absorbs them
        for b in mod.bundles:
            files, inline = fb.read_bundle_region(b.region)
            ebx, res, chunks, meta = fb.read_binary_bundle(inline) if inline else g.manifest(files, mod.root)
            first = 0 if inline else 1
            files = [arch.remap(mod, f) for f in files]
            copies.setdefault(b.name, []).append(dict(
                mod=mod, files=files, first=first, inline=inline, ebx=ebx, res=res, chunks=chunks,
                meta=fb.read_db(meta, 0)[0] if meta else None, load_flag=b.load_flag))

    def base_manifest(bundle_name):
        if bundle_name not in base:
            return None
        files, inline = fb.read_bundle_region(base[bundle_name].region)
        ebx, res, chunks, meta = g.manifest(files)
        return dict(files=files, first=1, ebx=ebx, res=res, chunks=chunks, meta=fb.read_db(meta, 0)[0] if meta else None)

    def payload(file_info):
        return fb.decode_cas(read_patch(file_info), g.root) if file_info.patch else g.payload(file_info)

    patch_cache = {}

    def read_patch(f):
        key = (f.install_chunk, f.archive)
        if key not in patch_cache:
            src = [s for s, ic, n in arch.sources if (ic, n) == key]
            patch_cache[key] = read(src[0]) if src else bytes(arch.own)
        return patch_cache[key][f.offset:f.offset + f.size]

    # ---- one version of every EBX/RES asset across the pack (by name), as ReSkate does
    bases = {n: base_manifest(n) for n in copies}
    base_sha, base_file = {}, {}
    for n, bm in bases.items():
        if bm:
            for i, a in enumerate(bm['ebx'] + bm['res']):
                base_sha[(a.kind, a.name)] = a.sha1
                base_file[(a.kind, a.name)] = (a, bm['files'][1 + i])
    versions = {}                                  # key -> {sha1: (asset, file, mod)}, highest priority last
    for n, cs in copies.items():
        for c in cs:
            for i, a in enumerate(c['ebx'] + c['res']):
                versions.setdefault((a.kind, a.name), {})[a.sha1] = (a, c['files'][c['first'] + i], c['mod'])
    final = {}
    for key, vs in versions.items():
        changed = {s: v for s, v in vs.items() if s != base_sha.get(key)}
        if not changed:
            continue
        if len(changed) == 1:
            final[key] = next(iter(changed.values()))[:2]
            continue
        order = list(changed.values())             # dict keeps insertion: lowest priority first
        kind, aname = key
        try:
            if kind == 'ebx' and key in base_sha:
                ba, bf = base_file[key]
                data = ebx_merge(g.payload(bf), [payload(f) for _, f, _ in order], work)
                f, sha = arch.add(data)
                a = fb.Asset(kind='ebx', name=aname, sha1=sha, original_size=len(data))
                final[key] = (a, f)
                notes.append(f'merged {len(order)} mods\' edits of {aname.split("/")[-1]}')
                continue
            if kind == 'res' and aname == BUNDLE_REF_TABLE and key in base_sha:
                ba, bf = base_file[key]
                data, meta, added, skipped = table_merge(g.payload(bf), ba.res_meta,
                                                         [(payload(f), a.res_meta) for a, f, _ in order])
                f, sha = arch.add(data)
                a = fb.Asset(kind='res', name=aname, sha1=sha, original_size=len(data), res_type=ba.res_type,
                             res_meta=meta, res_id=ba.res_id)
                final[key] = (a, f)
                notes.append(f'bundle-reference table: {added} rows from {len(order)} mods' +
                             (f' ({len(skipped)} skipped, e.g. {skipped[0]})' if skipped else ''))
                continue
        except Exception as e:
            notes.append(f'could not merge {aname}: {e}; kept {order[-1][2].name}\'s copy')
        final[key] = order[-1][:2]
        notes.append(f'{aname.split("/")[-1]}: changed differently by {", ".join(v[2].name for v in order)}; '
                     f'kept {order[-1][2].name}\'s copy')

    # ---- bundles
    out_bundles = []
    for n, cs in copies.items():
        bm = bases[n]
        single = len(cs) == 1
        c0 = cs[0]
        touched = any(final.get((a.kind, a.name), (a, None))[0].sha1 != a.sha1 for a in c0['ebx'] + c0['res'])
        if single and not touched:
            out_bundles.append(fb.TocBundle(n, fb.write_bundle_region(c0['files'], c0['inline']), c0['load_flag']))
            continue
        # one manifest: the game's assets in their order, then what each mod adds (priority order)
        order_keys, entry = [], {}
        layers = ([dict(bm, mod=None)] if bm else []) + cs
        for layer in layers:
            for i, a in enumerate(layer['ebx'] + layer['res']):
                key = (a.kind, a.name)
                if key not in entry:
                    order_keys.append(key)
                entry[key] = (a, layer['files'][layer['first'] + i])
        for key in order_keys:
            if key in final:
                entry[key] = final[key]
        chunk_order, chunk_entry = [], {}
        for layer in layers:
            off = layer['first'] + len(layer['ebx']) + len(layer['res'])
            for i, a in enumerate(layer['chunks']):
                if a.guid not in chunk_entry:
                    chunk_order.append(a.guid)
                elif chunk_entry[a.guid][0].sha1 != a.sha1 and layer.get('mod'):
                    notes.append(f'{n.split("/")[-1]}: chunk {a.name} differs between mods; kept {layer["mod"].name}\'s')
                chunk_entry[a.guid] = (a, layer['files'][off + i])
        # chunk metadata: the game's entries as they are, then what each mod adds over them. The
        # same entry may rightly appear many times (every LOD of a mesh has one), so a mod's
        # additions are its entries minus the game's, counted, not deduplicated.
        tree = None
        for layer in layers:
            if not layer['meta']:
                continue
            if tree is None:
                tree = dict(layer['meta'], children=list(layer['meta']['children']))
                ground = collections.Counter(fb.write_db(c) for c in tree['children'])
                continue
            left = collections.Counter(ground)
            for child in layer['meta']['children']:
                blob = fb.write_db(child)
                if left[blob] > 0:
                    left[blob] -= 1
                else:
                    tree['children'].append(child)
        ebx = [entry[k] for k in order_keys if k[0] == 'ebx']
        res = [entry[k] for k in order_keys if k[0] == 'res']
        chunks = [chunk_entry[gd] for gd in chunk_order]
        manifest = fb.write_binary_bundle([a for a, _ in ebx], [a for a, _ in res], [a for a, _ in chunks],
                                          fb.write_db(tree) if tree else b'')
        f = fb.FileInfo(True, arch.own_chunk, arch.own_archive, len(arch.own), len(manifest))
        arch.own += manifest                       # bundle manifests are stored raw, like the game's
        files = [f] + [x for _, x in ebx] + [x for _, x in res] + [x for _, x in chunks]
        out_bundles.append(fb.TocBundle(n, fb.write_bundle_region(files), cs[-1]['load_flag']))
        if tree and len(tree['children']) != len(chunks):
            notes.append(f'{n.split("/")[-1]}: {len(chunks)} chunks but {len(tree["children"])} metadata entries')

    # ---- items.toc chunks (highest priority last)
    toc_chunks = {}
    for mod in reversed(mods):
        for c in mod.chunks:
            if c.patch and not c.removed:
                c = fb.TocChunk(c.guid, True, c.install_chunk, arch.map[(mod.name, c.install_chunk, c.archive)], c.offset, c.size)
            if c.guid in toc_chunks and (toc_chunks[c.guid].offset, toc_chunks[c.guid].size) != (c.offset, c.size):
                notes.append(f'chunk {fb.guid_str(c.guid)} is in more than one mod; kept {mod.name}\'s')
            toc_chunks[c.guid] = c
    toc = fb.write_patch_toc(out_bundles, list(toc_chunks.values()), 3)

    # ---- the mod folder
    if os.path.exists(mod_dir):
        shutil.rmtree(mod_dir)
    os.makedirs(os.path.join(mod_dir, 'Win32'))
    arch.write(mod_dir)
    with open(os.path.join(mod_dir, 'Win32', 'items.toc'), 'wb') as f:
        f.write(toc)
    layout = read(os.path.join(g.root, 'Data', 'layout.toc'))
    with open(os.path.join(mod_dir, 'layout.toc'), 'wb') as f:
        f.write(layout[:4] + bytes(fb.TOC_ENVELOPE - 4) + layout[fb.TOC_ENVELOPE:])
    shutil.copyfile(os.path.join(g.root, 'Data', 'initfs_Win32'), os.path.join(mod_dir, 'initfs_win32'))
    skate_sha = hashlib.sha256(read(os.path.join(g.root, 'Skate.exe'))).hexdigest()
    with open(os.path.join(mod_dir, '.reskate-studio-patch'), 'w', newline='\n') as f:
        f.write(f'ReSkate Studio native Patch v1\nskate_sha256={skate_sha}\n')
    with open(os.path.join(mod_dir, 'manifest.json'), 'w') as f:
        json.dump({'author': author, 'dependencies': [], 'name': package, 'version_number': '1.0.0',
                   'description': f'{name}: ' + ', '.join(m.name for m in mods), 'website_url': ''}, f, indent=4)
    with open(os.path.join(mod_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(f'# {name}\n\nA pack of {len(mods)} mods, merged into one:\n\n')
        for m in mods:
            f.write(f'* {m.name}\n')
        for m in mods:
            if m.readme.strip():
                f.write(f'\n---\n\n## From {m.name}\n\n{m.readme.strip()}\n')
    icons = [Image.open(m.icon).convert('RGBA').resize((256, 256)) for m in mods if m.icon]
    sheet = Image.new('RGBA', (256, 256), (24, 24, 28, 255))
    if icons:
        n = int(np.ceil(np.sqrt(len(icons)))) if len(icons) > 1 else 1
        cell = 256 // n
        for i, im in enumerate(icons[:n * n]):
            sheet.alpha_composite(im.resize((cell, cell), Image.LANCZOS), ((i % n) * cell, (i // n) * cell))
    sheet.save(os.path.join(mod_dir, 'icon.png'))
    zip_path = os.path.join(out_root, f'{folder}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, files in os.walk(mod_dir):
            for fn in files:
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.relpath(full, mod_dir))
    for n_ in notes:
        log('  ' + n_)
    log(f'  wrote {mod_dir}\n  wrote {zip_path}')
    return mod_dir



# ------------------------------------------------------------------ mods.json
def mods_json(game):
    return os.path.join(game, 'Mods', 'mods.json')


def install(game, mod_dir, members, log=print):
    """Copies the pack into Mods and turns the mods it contains off (they stay installed)."""
    running = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Skate.exe'], capture_output=True, text=True).stdout
    if 'skate.exe' in running.lower():
        log('  skate. is running: close it, then run this again to install the pack.')
        return
    target = os.path.join(game, 'Mods', os.path.basename(mod_dir))
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(mod_dir, target)
    path = mods_json(game)
    data = json.load(open(path, encoding='utf-8')) if os.path.isfile(path) else {'mods': [], 'schema': 1}
    names = [m['name'] for m in data['mods']]
    for m in data['mods']:
        if m['name'] in members:
            m['enabled'] = False
    if os.path.basename(target) not in names:
        data['mods'].append({'enabled': True, 'name': os.path.basename(target)})
    else:
        for m in data['mods']:
            if m['name'] == os.path.basename(target):
                m['enabled'] = True
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2)
    log(f'  installed to {target}; turned off: {", ".join(members)}')


def installed_cosmetic_mods(g, game, skip=()):
    out = []
    mods = os.path.join(game, 'Mods')
    order = [m['name'] for m in json.load(open(mods_json(game), encoding='utf-8'))['mods']] if os.path.isfile(mods_json(game)) else []
    names = sorted((d for d in os.listdir(mods) if not d.startswith('.') and os.path.isdir(os.path.join(mods, d))),
                   key=lambda d: (order.index(d) if d in order else len(order), d))
    for d in names:
        if d in skip:
            continue
        try:
            out.append(Mod(g, os.path.join(mods, d)))
        except SystemExit:
            pass
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mods', nargs='*', help='installed mod folder names, highest priority first')
    p.add_argument('--name', help='pack name')
    p.add_argument('--author', default='socioculture')
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('--out', default=os.path.join(HERE, '..', 'mods'))
    p.add_argument('--interactive', action='store_true', help='choose the mods and name from a list')
    p.add_argument('--install', action='store_true', help='install the pack and turn its mods off in mods.json')
    p.add_argument('--no-verify', action='store_true')
    a = p.parse_args()
    g = fb.GameData(a.game)
    if a.interactive:
        packs = {d for d in os.listdir(os.path.join(a.game, 'Mods')) if d.startswith(a.author + '-')
                 and os.path.isfile(os.path.join(a.game, 'Mods', d, 'manifest.json'))
                 and 'merged into one' in open(os.path.join(a.game, 'Mods', d, 'README.md'), encoding='utf-8', errors='replace').read()}
        choices = installed_cosmetic_mods(g, a.game, packs)
        if not choices:
            raise SystemExit('No installed cosmetic mods to pack.')
        print('Installed cosmetic mods (maps cannot be packed):')
        for i, m in enumerate(choices, 1):
            print(f'  {i:2d}. {m.name}')
        pick = input('Numbers of the mods to pack, highest priority first (e.g. 1 3 4, or "all"): ').strip()
        idx = range(1, len(choices) + 1) if pick.lower() == 'all' else [int(x) for x in pick.replace(',', ' ').split()]
        mods = [choices[i - 1] for i in idx]
        a.name = input('Pack name: ').strip() or 'Modpack'
        if not a.install:
            a.install = input('Install it now and turn the packed mods off? (y/n): ').strip().lower().startswith('y')
    else:
        if not a.mods or not a.name:
            p.error('give the mod folder names and --name, or use --interactive')
        mods = [Mod(g, os.path.join(a.game, 'Mods', m)) for m in a.mods]
    if len(mods) < 2:
        raise SystemExit('Pick at least two mods.')
    print(f'Pack "{a.name}": {len(mods)} mods')
    mod_dir = build_pack(g, mods, a.name, a.author, os.path.abspath(a.out))
    if not a.no_verify:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'verify_modpack.py'), mod_dir, *[m.name for m in mods],
                            '--game', a.game], capture_output=True, text=True)
        print(r.stdout[-4000:])
        if r.returncode:
            print(r.stderr)
            raise SystemExit('Verification failed; the pack was not installed.')
    if a.install:
        if is_skate_running() or is_launcher_running():
            print('skate. or the ReSkate launcher is open: close both, then run this again to install '
                  '(or copy the folder into Mods yourself and turn the packed mods off in the launcher).')
        else:
            install(a.game, mod_dir, [m.name for m in mods])


if __name__ == '__main__':
    main()
