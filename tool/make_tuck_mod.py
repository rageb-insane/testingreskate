"""Tucks the game's pants into the high-tops built by make_shoe_mod: a ReSkate mod that gives each
pair of pants (and leggings, overalls) the region culling data the game gives only the body.

    python make_tuck_mod.py [--install]

skate. hides parts of a mesh by region: each item lists CulledRegions, and a mesh whose morph
preset (DingoMorphPreset) carries RegionCullingData drops the triangles tagged with those regions.
The game uses it only on the body (a shoe culls the bare foot), so pants clip through slim shoes.
This mod gives every pair of pants that data: the triangles under a high-top's collar (1.2 cm
below the rim of the game's Nike SB Dunk High, which make_shoe_mod fits high-tops to) are tagged
TUCK_REGION, the rest PANTS_REGION. High-tops built by make_shoe_mod cull TUCK_REGION, so with
them the pants end inside the shoe; every other shoe culls neither, so pants look as before.
A high-top placed as modelled has a collar of its own, not the Dunk's: make_shoe_mod leaves its
rim in mods/<mod>-tuck.json, and the pants triangles wholly under that rim are its own set. A
mods/<mod>-tuck.json of {"set": "dunk"} marks a shoe that hides the Dunk High set instead (the
Dior). A triangle can be in one region only, so the regions are the overlaps of those sets: each
triangle goes to the region of exactly the shoes that hide it ("dunk" alone keeps TUCK_REGION;
the others are "Socioculture_Tuck_" + their names joined by "+"), and every tucking shoe's item
in mods/ is then updated to cull every region its own set takes part in (tuck_items).

The culling data (resource type 0xC631492B, read from the body's) is: u8 version 2, u8 0,
u8 lods, u8 lods; u32 offset of the LOD table (0x28); u32 regions; u32 0; per LOD u8 lod, lod,
1, 0; per LOD u32 (lod | lod << 8), data offset, triangles; per region u32 id, triangles over
all LODs; per LOD per triangle (in the mesh's own order) u16 x3 vertex indices, u8 region, u8 0;
then 24 zero bytes. res_meta = (2, size).
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import meshset  # noqa: E402
import shoe_mesh  # noqa: E402
from make_deck_mod import DEFAULT_GAME, PATCH_DIRECTORY, ebx_info, edit_ebx, new_guid, CasWriter  # noqa: E402
from make_shoe_mod import bundles_of_other_mods, earlier_builds, ebx_file_guid  # noqa: E402

BODY_BUNDLE = ('win32/characters/maincharacters/generic/cas/body/bodyblends/rsp/'
               'cas_rsp_body_complex_dmpreset_cas_main_bundlereftable')
CULL_RES = 0xC631492B
FOOT_REGION = 0x7C7EB057      # the body's region for both feet, up to 13.3 cm (the VertClassic hides it)
FOOT_PART_PREFIX = 'Socioculture_Foot_'


BODY_REGION_GROUPS = os.path.join(HERE, 'body_region_groups.json')   # see body_region_groups()


def body_region_groups(g, by_name, ids):
    """Sets of body regions the game's items always hide together (every item that lists one lists
    the others): merging such regions changes nothing, and frees their slots. The body has 32
    regions, which looks like the game's limit (one bit each), so new ones must reuse slots.
    Scans every item for the 32-bit region ids; cached in body_region_groups.json."""
    if os.path.isfile(BODY_REGION_GROUPS):
        with open(BODY_REGION_GROUPS) as f:
            cached = json.load(f)
        if cached.get('ids') == ids:
            return cached['same']
    _, bundles, _ = g.toc('Win32/items.toc')
    used = {rid: set() for rid in ids}
    for b in bundles:
        if not (b.name.endswith('sharedbundle') or '/items/' in b.name):
            continue
        try:
            fs, _ = fb.read_bundle_region(b.region)
            ee = g.manifest(fs)[0]
        except Exception:
            continue
        for a, f in zip(ee, fs[1:]):
            if not a.name.startswith('items/'):
                continue
            try:
                raw = g.payload(f)
            except Exception:
                continue
            for rid in ids:
                if struct.pack('<I', rid) in raw:
                    used[rid].add(a.name)
    groups = {}
    for k, rid in enumerate(ids):
        groups.setdefault(frozenset(used[rid]), []).append(k)
    same = [v for v in groups.values() if len(v) > 1]
    with open(BODY_REGION_GROUPS, 'w') as f:
        json.dump({'ids': ids, 'same': same}, f)
    return same


def body_feet(g, by_name, toc, cas, low_shoes, log):
    """Splits the body's foot region along the collars of the shoes that keep the bare feet
    drawn (their collars dip under its top, 13.3 cm: left whole, the ankle above them would be
    missing; left alone, the heel and ankle push through them). Each foot triangle goes to the
    part of exactly the low shoes it is wholly under; the part under all of them keeps the foot
    region's own id, so the game's VertClassic (and every shoe that hides the feet) still hides
    the foot; the rest are "Socioculture_Foot_" + names. The body keeps its 32 regions: a part
    takes the slot of a region merged into one the game always hides with it. Parts beyond the
    free slots fold into the part above them (drawn with fewer shoes). low_shoes: [(name, rims)].
    Returns (the body bundle for the mod, [(names, region id)] of the parts)."""
    from make_deck_mod import djb
    files, _ = fb.read_bundle_region(by_name[BODY_BUNDLE].region)
    e, r, c, meta = g.manifest(files)
    ne, nr = len(e), len(r)
    f_ebx, f_res, f_ch = list(files[1:1 + ne]), list(files[1 + ne:1 + ne + nr]), files[1 + ne + nr:]
    ri = next(i for i, a in enumerate(r) if a.res_type == CULL_RES)
    mi = next(i for i, a in enumerate(r) if a.res_type == shoe_mesh.MESHSET_RES)
    data = g.payload(f_res[ri])
    n_lods, n_reg = data[2], struct.unpack_from('<I', data, 8)[0]
    regions = [struct.unpack_from('<I', data, 0x28 + 12 * n_lods + 8 * k)[0] for k in range(n_reg)]
    foot = regions.index(FOOT_REGION)
    merge = {}                                           # slot -> the slot whose region it joins
    for group in body_region_groups(g, by_name, regions):
        for k in group[1:]:
            merge[k] = group[0]
    ms = meshset.MeshSet(g.payload(f_res[mi]))
    names = [n for n, _ in low_shoes]
    full = (1 << len(low_shoes)) - 1
    lods, counts = [], {}
    for k, lod in enumerate(ms.lods):
        _, off, n = struct.unpack_from('<III', data, 0x28 + 12 * k)
        rows = np.frombuffer(data, np.uint16, n * 4, off).reshape(-1, 4)
        tri, reg = rows[:, :3].astype(np.int64), rows[:, 3].astype(np.int64)
        pos = meshset.decode_section(lod, lod.sections[0], lod_chunk(g, toc, lod, list(zip(c, f_ch))))['pos'][:, :3]
        code = np.zeros(len(tri), np.int64)
        for bit, (_, rims) in enumerate(low_shoes):
            code |= tucked(pos, tri, rims, 0.0).astype(np.int64) << bit
        code[reg != foot] = -1
        for cd in np.unique(code[code >= 0]):
            counts[int(cd)] = counts.get(int(cd), 0) + int((code == cd).sum())
        lods.append((tri, reg, code))
    # the parts that need a slot, biggest first; the rest fold into the part with fewest shoes
    # among those that hide a subset of theirs (drawn with fewer shoes: never a hole)
    parts = sorted((cd for cd in counts if cd != full), key=lambda cd: -counts[cd])
    slots = sorted(merge)
    if len(parts) > len(slots):
        log(f'  body: {len(parts)} foot parts but {len(slots)} free region slots; the smallest are drawn with fewer shoes')
    kept = parts[:len(slots)]
    fold = {}
    for cd in parts[len(slots):]:
        subsets = [k for k in kept + [0] if (k & cd) == k]
        fold[cd] = max(subsets, key=lambda k: bin(k).count('1'))
    new_regions = list(regions)
    slot_of = {}
    for cd, slot in zip(kept, slots):
        slot_of[cd] = slot
        new_regions[slot] = djb(FOOT_PART_PREFIX + ('+'.join(nm for b, nm in enumerate(names) if cd >> b & 1) or 'none'))
    out = []
    for tri, reg, code in lods:
        reg = reg.copy()
        for k_from, k_to in merge.items():
            reg[reg == k_from] = k_to
        for cd in parts:
            target = fold.get(cd, cd)
            reg[code == cd] = slot_of[target] if target in slot_of else foot if target == full else slot_of.get(0, foot)
        out.append((tri, reg.astype(np.uint16)))
    new = cull_data(out, new_regions)
    fi, sha = cas.add(new)
    r[ri].sha1, r[ri].original_size, f_res[ri] = sha, len(new), fi
    r[ri].res_meta = struct.pack('<IIII', 2, len(new), 0, 0)
    manifest = fb.write_binary_bundle(e, r, c, meta)
    bundle = fb.TocBundle(BODY_BUNDLE, fb.write_bundle_region([cas.add_raw(manifest)] + f_ebx + f_res + list(f_ch)))
    every = [(tuple(names), FOOT_REGION)] + [(tuple(nm for b, nm in enumerate(names) if cd >> b & 1), new_regions[slot])
                                             for cd, slot in slot_of.items()]
    log(f'  body: foot region split along the collars of {", ".join(names)} into {len(slot_of) + 1} parts, '
        f'still {len(new_regions)} regions ({len(merge)} merged with regions the game always hides together)')
    return bundle, every
PANTS = ('_pants_', '_overalls_', '_leggings')


def read(path):
    with open(path, 'rb') as f:
        return f.read()


def cull_data(lods, regions):
    """lods: [(triangles (n, 3), region index per triangle)]; regions: [region id]."""
    n = len(lods)
    head = bytearray([2, 0, n, n]) + struct.pack('<III', 0x28, len(regions), 0)
    head += b''.join(bytes([k, k, 1, 0]) for k in range(n))
    table_at = len(head)
    data_at = table_at + 12 * n + 8 * len(regions)
    lod_table, body, at = b'', b'', data_at
    for k, (tri, reg) in enumerate(lods):
        rows = np.zeros((len(tri), 4), np.uint16)
        rows[:, :3] = tri
        rows[:, 3] = reg
        lod_table += struct.pack('<III', k | k << 8, at, len(tri))
        body += rows.tobytes()
        at += len(tri) * 8
    counts = [sum(int((reg == i).sum()) for _, reg in lods) for i in range(len(regions))]
    region_table = b''.join(struct.pack('<II', rid, c) for rid, c in zip(regions, counts))
    assert len(head) == 0x28
    return bytes(head) + lod_table + region_table + body + bytes(24)


def lod_chunk(g, toc, lod, bundle_chunks):
    """A LOD's vertex/index chunk: from the TOC, else whole in its bundle."""
    ck = toc.get(lod.chunk)
    if ck is not None:
        return fb.decode_cas(g.read(fb.FileInfo(ck.patch, ck.install_chunk, ck.archive, ck.offset, ck.size)), g.root)
    hit = [(x, f) for x, f in bundle_chunks if x.guid == lod.chunk and x.logical_offset == 0]
    if not hit:
        raise SystemExit(f'mesh chunk {fb.guid_str(lod.chunk)} not found')
    return g.payload(hit[0][1])


def tucked(positions, triangles, rims, below=shoe_mesh.TUCK_BELOW_RIM):
    """Per triangle: under the collar rim of the high-top on its foot (`below` metres under it)."""
    centre = positions[triangles].mean(1)
    out = np.zeros(len(triangles), bool)
    for side, (axis, rim) in rims.items():
        on = centre[:, 0] * side > 0
        v = centre[on][:, [0, 2]] - axis
        sector = ((np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360) // (360 // len(rim))).astype(int)
        top = positions[triangles[on]][:, :, 1].max(1)
        out[on] = top < rim[sector] - below
    return out


DUNK_SET = 'dunk'


def tucking_shoes(out_root):
    """The tucking high-tops built into `out_root`: [(mod folder, set name, rims or None, hides the
    bare feet)], from the
    <mod>-tuck.json files beside them. rims None: the shoe hides the Dunk High set."""
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(out_root, '*-tuck.json'))):
        with open(path) as f:
            j = json.load(f)
        folder = os.path.basename(path)[:-len('-tuck.json')]
        feet = j.get('hide_feet', True)
        if j.get('set') == DUNK_SET:
            out.append((folder, DUNK_SET, None, feet))
            continue
        rims = {int(k): (np.array(v['axis']), np.array(v['rim'])) for k, v in j['rims'].items()}
        out.append((folder, j['name'][len(shoe_mesh.OWN_TUCK_PREFIX):], rims, feet))
    return out


def region_id(names):
    """The region of the triangles exactly these sets hide."""
    if not names:
        return shoe_mesh.PANTS_REGION
    if list(names) == [DUNK_SET]:
        return shoe_mesh.TUCK_REGION
    from make_deck_mod import djb
    return djb(shoe_mesh.OWN_TUCK_PREFIX + '+'.join(sorted(names)))


def build(args):
    game = args.game
    author = re.sub(r'[^A-Za-z0-9_]', '_', args.author).strip('_') or 'socioculture'
    package = 'High_Top_Pants_Tuck'
    folder_name = f'{author}-{package}'
    out_root = os.path.abspath(args.out)
    mod_dir = os.path.join(out_root, folder_name)
    rng = random.SystemRandom()
    print(f'Pants tuck  mod={folder_name}')
    g = fb.GameData(game)
    patch_chunk = next(k for k, v in g.chunk_dirs.items() if v == PATCH_DIRECTORY)
    _, bundles, toc_chunks = g.toc('Win32/items.toc')
    by_name = {b.name: b for b in bundles}
    toc = {c.guid: c for c in toc_chunks}
    shape = shoe_mesh.Reference(g, shoe_mesh.DUNK_HIGH_BUNDLE, None)
    rims = {side: shoe_mesh.collar_rim(shape, side) for side in (1, -1)}
    print('  Dunk High collar rim: %.1f-%.1f cm (tucking up to %.1f cm under it)'
          % (rims[1][1].min() * 100, rims[1][1].max() * 100, shoe_mesh.TUCK_BELOW_RIM * 100))
    shoes = tucking_shoes(out_root)
    # the Dunk High set only when a shoe that hides it is registered (a mods/<mod>-tuck.json of {"set": "dunk"})
    sets = ([DUNK_SET] if any(r is None for _, _, r, _ in shoes) else []) + sorted({name for _, name, r, _ in shoes if r is not None})
    if not sets:
        raise SystemExit('No tucking high-tops in ' + out_root + ' (no <mod>-tuck.json): nothing to tuck pants into.')
    own_rims = {name: r for _, name, r, _ in shoes if r is not None}
    for name in [x for x in sets if x != DUNK_SET]:
        r = own_rims[name]
        print('  %s: its own collar rim %.1f-%.1f cm (pants wholly under it are hidden with that shoe)'
              % (name, r[1][1].min() * 100, r[1][1].max() * 100))
    others = bundles_of_other_mods(game, folder_name)

    work = tempfile.mkdtemp(prefix='tuckmod_')
    try:
        files, _ = fb.read_bundle_region(by_name[BODY_BUNDLE].region)
        b_ebx = g.manifest(files)[0]
        template = next(g.payload(f) for a, f in zip(b_ebx, files[1:]) if a.name.endswith('_regioncullingdata'))
        tpath = os.path.join(work, 'cull_template.ebx')
        with open(tpath, 'wb') as f:
            f.write(template)

        cas = CasWriter(patch_chunk, game, compress=not args.no_compress)
        out_bundles, report, staged = [], [], []
        for name in sorted(by_name):
            leaf = name.split('/')[-1].replace('_complex_dmpreset_cas_main_bundlereftable', '')
            if not name.endswith('_complex_dmpreset_cas_main_bundlereftable') or not any(k in leaf for k in PANTS):
                continue
            if name in others:
                print(f'  skipped {leaf}: also shipped by {", ".join(others[name])}')
                continue
            files, _ = fb.read_bundle_region(by_name[name].region)
            e, r, c, meta = g.manifest(files)
            ne, nr = len(e), len(r)
            f_ebx, f_res, f_ch = list(files[1:1 + ne]), list(files[1 + ne:1 + ne + nr]), files[1 + ne + nr:]
            di = next((i for i, a in enumerate(e) if a.name.endswith('_complex_dmpreset')), None)
            mi = next((i for i, a in enumerate(r) if a.res_type == shoe_mesh.MESHSET_RES), None)
            if di is None or mi is None:
                print(f'  skipped {leaf}: no morph preset or mesh')
                continue
            dm_path = os.path.join(work, leaf + '_dm.ebx')
            with open(dm_path, 'wb') as f:
                f.write(g.payload(f_ebx[di]))
            if 'RegionCullingData = null' not in ebx_info(dm_path)['dump']:
                print(f'  skipped {leaf}: it already has culling data')
                continue
            ms = meshset.MeshSet(g.payload(f_res[mi]))
            if any(len(lod.sections) != 1 for lod in ms.lods):
                print(f'  skipped {leaf}: a mesh part layout this tool does not handle')
                continue
            lods = []
            for k, lod in enumerate(ms.lods):
                chunk = lod_chunk(g, toc, lod, list(zip(c, f_ch)))
                v = meshset.decode_section(lod, lod.sections[0], chunk)
                tri = v['triangles']
                pos = v['pos'][:, :3]
                code = np.zeros(len(tri), np.int64)        # which sets hide each triangle, a bit per set
                for bit, set_name in enumerate(sets):
                    hidden = (tucked(pos, tri, rims) if set_name == DUNK_SET
                              else tucked(pos, tri, own_rims[set_name], 0.0))
                    code |= hidden.astype(np.int64) << bit
                lods.append((tri, code))
            staged.append((name, leaf, e, r, c, meta, f_ebx, f_res, f_ch, di, dm_path, lods))
        # the regions: every combination of sets some triangle has (the same list for every pair)
        codes = sorted({int(x) for *_, lods in staged for _, code in lods for x in np.unique(code)})
        if 0 not in codes:
            codes = [0] + codes
        combos = [tuple(sets[b] for b in range(len(sets)) if code >> b & 1) for code in codes]
        regions = [region_id(names) for names in combos]
        index = np.full(max(codes) + 1, -1, np.int64)
        index[codes] = np.arange(len(codes))
        print('  regions: ' + ', '.join(('+'.join(names) or 'the rest') for names in combos))
        for name, leaf, e, r, c, meta, f_ebx, f_res, f_ch, di, dm_path, lods in staged:
            hidden0 = {s: int(((lods[0][1] >> b) & 1).sum()) for b, s in enumerate(sets)}
            lods = [(tri, index[code].astype(np.uint16)) for tri, code in lods]
            data = cull_data(lods, regions)
            base = e[di].name[:-len('_complex_dmpreset')]
            cull_name = base + '_regioncullingdata'
            res_name = cull_name + cull_name + '_basemesh_cullmask_primitives'
            ids = {'file': new_guid(), 'inst': new_guid()}
            res_id = rng.getrandbits(64) | 1
            cull_ebx = edit_ebx(tpath, os.path.join(work, leaf + '_cull.ebx'), '--file-guid', ids['file'],
                                '--inst-guid', 0, ids['inst'], '--set', '0:Name', 'str:' + cull_name,
                                '--set', '0:CullingPrimitiveResource', f'res:{res_id}')
            dm = edit_ebx(dm_path, os.path.join(work, leaf + '_dm_new.ebx'),
                          '--set-import', '0:RegionCullingData', ids['file'], ids['inst'])
            fi, sha = cas.add(dm)
            e[di].sha1, e[di].original_size, f_ebx[di] = sha, len(dm), fi
            fi_e, sha_e = cas.add(cull_ebx)
            fi_r, sha_r = cas.add(data)
            new_e = fb.Asset(kind='ebx', name=cull_name, sha1=sha_e, original_size=len(cull_ebx))
            new_r = fb.Asset(kind='res', name=res_name, sha1=sha_r, original_size=len(data), res_type=CULL_RES,
                             res_meta=struct.pack('<IIII', 2, len(data), 0, 0), res_id=res_id)
            manifest = fb.write_binary_bundle(e + [new_e], r + [new_r], c, meta)
            out_bundles.append(fb.TocBundle(name, fb.write_bundle_region(
                [cas.add_raw(manifest)] + f_ebx + [fi_e] + f_res + [fi_r] + list(f_ch))))
            report.append((leaf, hidden0, len(lods[0][0])))
            print(f'  {leaf}: of {len(lods[0][0])} triangles, hidden under '
                  + ', '.join(f'{"the Dunk-shaped high-tops" if s == DUNK_SET else s} {n}' for s, n in hidden0.items()))
        # the body's feet, split for shoes that keep them drawn (their collars dip under the feet)
        low_shoes = [(name, r) for _, name, r, feet in shoes if r is not None and not feet]
        foot_parts = []
        if low_shoes:
            if BODY_BUNDLE in others:
                print(f'  body: also shipped by {", ".join(others[BODY_BUNDLE])}; its feet are left as they are')
            else:
                body_bundle, foot_parts = body_feet(g, by_name, toc, cas, low_shoes, print)
                out_bundles.append(body_bundle)
        if not out_bundles:
            raise SystemExit('No pants to change.')
        toc_bytes = fb.write_patch_toc(out_bundles, [], 3)

        if os.path.exists(mod_dir):
            shutil.rmtree(mod_dir)
        cas_dir = os.path.join(mod_dir, 'Win32', *PATCH_DIRECTORY.split('/'))
        os.makedirs(cas_dir)
        with open(os.path.join(cas_dir, 'cas_01.cas'), 'wb') as f:
            f.write(cas.data)
        with open(os.path.join(mod_dir, 'Win32', 'items.toc'), 'wb') as f:
            f.write(toc_bytes)
        layout = read(os.path.join(game, 'Data', 'layout.toc'))
        with open(os.path.join(mod_dir, 'layout.toc'), 'wb') as f:
            f.write(layout[:4] + bytes(fb.TOC_ENVELOPE - 4) + layout[fb.TOC_ENVELOPE:])
        shutil.copyfile(os.path.join(game, 'Data', 'initfs_Win32'), os.path.join(mod_dir, 'initfs_win32'))
        skate_sha = hashlib.sha256(read(os.path.join(game, 'Skate.exe'))).hexdigest()
        with open(os.path.join(mod_dir, '.reskate-studio-patch'), 'w', newline='\n') as f:
            f.write(f'ReSkate Studio native Patch v1\nskate_sha256={skate_sha}\n')
        with open(os.path.join(mod_dir, 'manifest.json'), 'w') as f:
            json.dump({'author': author, 'dependencies': [], 'description': 'Pants tuck into custom high-tops instead of clipping',
                       'name': package, 'version_number': args.version, 'website_url': ''}, f, indent=4)
        with open(os.path.join(mod_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write('# High-Top Pants Tuck\n\nWith custom high-top shoes made by the ReSkate Deck tools, the game\'s pants '
                    'end inside the shoe instead of clipping through it. Other shoes are unaffected.\n')
        zip_path = os.path.join(out_root, f'{folder_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for dirpath, _, fs in os.walk(mod_dir):
                for n in fs:
                    full = os.path.join(dirpath, n)
                    z.write(full, os.path.relpath(full, mod_dir))
        print(f'  {len(out_bundles)} pants changed\n  wrote {mod_dir}\n  wrote {zip_path}')

        problems = verify(g, mod_dir, cas)
        print('\nALL CHECKS PASSED' if not problems else '\n' + '\n'.join(problems))
        if problems:
            raise SystemExit('Verification failed; the mod was not installed.')

        # every tucking shoe's item culls the regions its own set takes part in
        import tuck_items
        changed = []
        for shoe_folder, set_name, r, feet in shoes:
            shoe_dir = os.path.join(out_root, shoe_folder)
            if not os.path.isdir(shoe_dir):
                print(f'  {shoe_folder}: not in {out_root}, its item was left alone')
                continue
            ids = [rid for rid, names in zip(regions, combos) if set_name in names]
            # the body's foot parts: a shoe that hides the feet hides every part, a low one those under it
            ids += [rid for names, rid in foot_parts if (feet and rid != FOOT_REGION) or (not feet and set_name in names)]
            before = tuck_items.item_regions(g, shoe_dir)
            after = tuck_items.set_item_regions(g, shoe_dir, ids, feet)
            if after != before:
                changed.append(shoe_folder)
                with zipfile.ZipFile(os.path.join(out_root, f'{shoe_folder}.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
                    for dirpath, _, fs in os.walk(shoe_dir):
                        for n in fs:
                            full = os.path.join(dirpath, n)
                            z.write(full, os.path.relpath(full, shoe_dir))
            check = subprocess.run([sys.executable, os.path.join(HERE, 'verify_shoe_mod.py'), shoe_dir, '--game', game]
                                   + ([] if r is None else ['--as-modelled']), capture_output=True, text=True)
            print(f'  {shoe_folder}: culls {len(ids)} pants region(s)'
                  + (' (item updated)' if shoe_folder in changed else ' (unchanged)')
                  + ('' if check.returncode == 0 else '  VERIFY FAILED:\n' + check.stdout[-1500:]))
            if check.returncode:
                raise SystemExit(f'{shoe_folder} failed verification after its item was updated.')
        if changed:
            print('  copy these into Mods along with the tuck mod: ' + ', '.join(changed))

        if args.install:
            running = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Skate.exe'], capture_output=True, text=True).stdout
            if 'skate.exe' in running.lower():
                print('  skate. is running: close it, then copy the mod folders into Mods (or run this again).')
            else:
                for old in earlier_builds(game, folder_name):
                    shutil.rmtree(os.path.join(game, 'Mods', old))
                for folder in [folder_name] + changed:
                    target = os.path.join(game, 'Mods', folder)
                    if os.path.exists(target):
                        shutil.rmtree(target)
                    shutil.copytree(os.path.join(out_root, folder), target)
                    print(f'  installed to {target}')
        return mod_dir, changed
    finally:
        shutil.rmtree(work, ignore_errors=True)


def verify(g, mod_dir, cas):
    """Reads the mod back: each changed bundle keeps the game's assets, its morph preset points
    at the new culling asset, which names the new resource, whose triangles are the mesh's own."""
    problems = []
    _, bundles, _ = fb.read_toc(read(os.path.join(mod_dir, 'Win32', 'items.toc')))
    _, base_bundles, toc_chunks = g.toc('Win32/items.toc')
    base = {b.name: b for b in base_bundles}
    toc = {c.guid: c for c in toc_chunks}
    for b in bundles:
        leaf = b.name.split('/')[-1][:40]
        files, _ = fb.read_bundle_region(b.region)
        e, r, c, meta = g.manifest(files, mod_dir)
        be, br, bc, _ = g.manifest(fb.read_bundle_region(base[b.name].region)[0])
        if [x.name for x in e[:len(be)]] != [x.name for x in be] or [x.name for x in r[:len(br)]] != [x.name for x in br]:
            problems.append(f'{leaf}: game assets not kept in place')
        pay = {}
        for x, f in zip(e + r, files[1:]):
            raw = bytes(cas.data[f.offset:f.offset + f.size]) if f.patch else None
            if raw is not None and fb.sha1(raw) != x.sha1:
                problems.append(f'{leaf}: {x.name} sha1 mismatch')
            pay[(x.kind, x.name)] = fb.decode_cas(raw, g.root) if raw is not None else g.payload(f)
        dm = next(v for (k, n), v in pay.items() if k == 'ebx' and n.endswith('_complex_dmpreset'))
        cull = next(v for (k, n), v in pay.items() if k == 'ebx' and n.endswith('_regioncullingdata'))
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ebx') as t:
            t.write(dm)
        dump = ebx_info(t.name)['dump']
        os.unlink(t.name)
        if ebx_file_guid(cull) not in re.search(r'RegionCullingData = (.*)', dump).group(1):
            problems.append(f'{leaf}: the morph preset does not point at its culling data')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ebx') as t:
            t.write(cull)
        rid = int(re.search(r'res\(0x([0-9a-f]+)\)', ebx_info(t.name)['dump']).group(1), 16)
        os.unlink(t.name)
        rx = [x for x in r if x.res_type == CULL_RES]
        if len(rx) != 1 or rx[0].res_id != rid:
            problems.append(f'{leaf}: culling resource missing or not the one named')
            continue
        data = pay[('res', rx[0].name)]
        if struct.unpack_from('<II', rx[0].res_meta)[1] != len(data):
            problems.append(f'{leaf}: res_meta size')
        mi = next(i for i, x in enumerate(r) if x.res_type == shoe_mesh.MESHSET_RES)
        ms = meshset.MeshSet(pay[('res', r[mi].name)])
        if data[2] != len(ms.lods):
            problems.append(f'{leaf}: LOD count')
        for k, lod in enumerate(ms.lods):
            key, off, n = struct.unpack_from('<III', data, 0x28 + 12 * k)
            chunk = lod_chunk(g, toc, lod, list(zip(c, files[1 + len(e) + len(r):])))
            tri = meshset.decode_section(lod, lod.sections[0], chunk)['triangles']
            rows = np.frombuffer(data, np.uint16, n * 4, off).reshape(-1, 4)
            n_regions = struct.unpack_from('<I', data, 8)[0]
            if n != len(tri) or not np.array_equal(rows[:, :3].astype(np.int64), tri) or rows[:, 3].max() >= n_regions:
                problems.append(f'{leaf}: LOD{k} triangles do not match the mesh')
        if not problems:
            print(f'  ok   {leaf}')
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--author', default='socioculture')
    p.add_argument('--version', default='1.0.0')
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('--out', default=os.path.join(HERE, '..', 'mods'))
    p.add_argument('--install', action='store_true')
    p.add_argument('--no-compress', action='store_true')
    build(p.parse_args())


if __name__ == '__main__':
    main()
