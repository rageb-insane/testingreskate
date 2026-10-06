"""Rewrites the culled regions of the item in a shoe mod already built into mods/<folder>, in place.

make_tuck_mod splits every pair of pants into regions by which tucking high-tops hide them (a
triangle can be in one region only, so the regions are the overlaps of the shoes' collars), and
each tucking shoe's item must list the regions under its own collar. Only the item changes: its
new EBX is appended to the mod's cas, the shared bundle's manifest points at it, and the
items.toc is written again; every other byte the mod ships stays where it was."""
import os
import re
import tempfile

import fb
from make_deck_mod import PATCH_DIRECTORY, SHARED_BUNDLE, ebx_info, edit_ebx

FOOT_REGIONS = (0x7C7EB057, 0xB17618AF)     # the bare foot, which every shoe hides


def _paths(mod_dir):
    return (os.path.join(mod_dir, 'Win32', *PATCH_DIRECTORY.split('/'), 'cas_01.cas'),
            os.path.join(mod_dir, 'Win32', 'items.toc'))


def _item(g, mod_dir):
    cas_path, toc_path = _paths(mod_dir)
    cas = bytearray(open(cas_path, 'rb').read())
    flags, bundles, chunks = fb.read_toc(open(toc_path, 'rb').read())
    k = next(i for i, b in enumerate(bundles) if b.name == SHARED_BUNDLE)
    files, inline = fb.read_bundle_region(bundles[k].region)
    e, r, c, meta = g.manifest(files, mod_dir)
    i = next(i for i, a in enumerate(e)                       # the mod's own item (make_shoe_mod names it own_<shoe>)
             if a.name.startswith('items/cust_shoes/own_') and files[1 + i].patch)
    f = files[1 + i]
    payload = fb.decode_cas(bytes(cas[f.offset:f.offset + f.size]), g.root)
    return cas, bundles, chunks, k, files, inline, (e, r, c, meta), i, payload


def _regions(payload):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ebx') as t:
        t.write(payload)
    try:
        dump = ebx_info(t.name)['dump']
    finally:
        os.unlink(t.name)
    block = dump[dump.index('CulledRegions'):]
    return [int(x, 16) for x in re.findall(r'RegionId = \d+ \(0x([0-9a-f]+)\)', block)]


def item_regions(g, mod_dir):
    """The region ids the mod's item culls."""
    return _regions(_item(g, mod_dir)[-1])


def set_item_regions(g, mod_dir, tuck_ids, feet=True):
    """Makes the item cull the bare feet (unless feet=False: a shoe whose collar dips under the
    feet's regions) plus `tuck_ids`. Entries left over from a longer list repeat the first (the
    EBX tool can add array entries but not drop them); nothing at all is region 0. Returns the ids."""
    cas, bundles, chunks, k, files, inline, (e, r, c, meta), i, payload = _item(g, mod_dir)
    want = (list(FOOT_REGIONS) if feet else []) + [int(x) for x in tuck_ids] or [0]
    have = _regions(payload)
    ops = []
    for _ in range(max(0, len(want) - len(have))):
        ops += ['--append', '0:ItemData.CulledRegions']
    final = want + [want[0]] * max(0, len(have) - len(want))
    for n, rid in enumerate(final):
        ops += ['--set', f'0:ItemData.CulledRegions[{n}].RegionId', f'u64:{rid}']
    work = tempfile.mkdtemp(prefix='tuckitem_')
    src, dst = os.path.join(work, 'item.ebx'), os.path.join(work, 'item_new.ebx')
    with open(src, 'wb') as fh:
        fh.write(payload)
    new = edit_ebx(src, dst, *ops)
    assert _regions(new) == final, 'item culled regions did not take'
    old = files[1 + i]
    encoded = fb.encode_cas(new, g.root, True)
    assert fb.decode_cas(encoded, g.root) == new
    files[1 + i] = fb.FileInfo(old.patch, old.install_chunk, old.archive, len(cas), len(encoded))
    cas += encoded
    e[i].sha1, e[i].original_size = fb.sha1(encoded), len(new)
    manifest = fb.write_binary_bundle(e, r, c, meta)
    files[0] = fb.FileInfo(files[0].patch, files[0].install_chunk, files[0].archive, len(cas), len(manifest))
    cas += manifest
    bundles[k] = fb.TocBundle(bundles[k].name, fb.write_bundle_region(files, inline))
    toc = fb.write_patch_toc(bundles, chunks, 3)
    cas_path, toc_path = _paths(mod_dir)
    with open(cas_path, 'wb') as fh:
        fh.write(cas)
    with open(toc_path, 'wb') as fh:
        fh.write(toc)
    return final
