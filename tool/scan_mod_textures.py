"""Scans every installed mod's textures for the header/chunk mismatches that crash
skate. when a texture streams in (wrong stream offsets, sizes that don't add up,
mips that don't start a compressed block, bundle copies that don't match).

    python scan_mod_textures.py [--game E:\\...\\Skate]
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
from make_deck_mod import DEFAULT_GAME, djb, TEXTURE_RES  # noqa: E402


def block_starts(raw):
    at, logical, starts = 0, 0, {}
    while at < len(raw):
        starts[logical] = at
        comp = struct.unpack_from('<H', raw, at + 4)[0]
        logical += fb.be32(raw, at) & 0xFFFFFF
        at += 8 + struct.unpack_from('>H', raw, at + 6)[0] + (((comp >> 8) & 0x0F) << 16)
    return starts, logical


def scan_mod(g, root):
    issues, count = [], 0
    for dirpath, _, files in os.walk(os.path.join(root, 'Win32')):
        for f in files:
            if not f.endswith('.toc'):
                continue
            try:
                _, bundles, chunks = fb.read_toc(open(os.path.join(dirpath, f), 'rb').read())
            except Exception as e:
                issues.append(f'{f}: unreadable TOC ({e})')
                continue
            toc = {c.guid: c for c in chunks}
            for b in bundles:
                try:
                    files_, _ = fb.read_bundle_region(b.region)
                    ebx, res, ch, _ = g.manifest(files_, root)
                except Exception:
                    continue
                bundle_chunks = {a.guid: (a, files_[1 + len(ebx) + len(res) + i]) for i, a in enumerate(ch)}
                for i, a in enumerate(res):
                    fi = files_[1 + len(ebx) + i]
                    if a.res_type != TEXTURE_RES or not fi.patch:
                        continue
                    count += 1
                    name = a.name.split('/')[-1][:60]
                    try:
                        h = g.payload(fi, root)
                        mips, first = h[30], h[31]
                        w, hh = struct.unpack_from('<HH', h, 22)
                        sizes = struct.unpack_from(f'<{mips}I', h, 56)
                        total = struct.unpack_from('<I', h, 116)[0]
                        guid = h[40:56]
                        tag = f'{name} ({w}x{hh}, {mips} mips, first {first})'
                        if a.res_id & 1 == 0:
                            issues.append(f'{tag}: resource id {a.res_id:#x} has bit 0 clear')
                        if sum(sizes) != total:
                            issues.append(f'{tag}: mip sizes add to {sum(sizes)}, header says {total}')
                        if len(h) >= 128 and struct.unpack_from('<Q', h, 120)[0] != djb(a.name, 64):
                            issues.append(f'{tag}: name hash does not match its name')
                        src = toc.get(guid)
                        if src is None and guid not in bundle_chunks:
                            issues.append(f'{tag}: its chunk is in neither the TOC nor the bundle (base game?)')
                            continue
                        if src is not None:
                            loc = fb.FileInfo(src.patch, src.install_chunk, src.archive, src.offset, src.size)
                            raw = g.read(loc, root) if src.patch else g.read(loc)
                            starts, logical = block_starts(raw)
                            if logical != total:
                                issues.append(f'{tag}: streamed chunk holds {logical} bytes, header says {total}')
                            mip_start = [sum(sizes[:k]) for k in range(mips)]
                            want = (starts.get(mip_start[1], -1) if mips > 1 else 0,
                                    starts.get(mip_start[2], -1) if mips > 2 else 0)
                            have = struct.unpack_from('<II', h, 0)
                            if mips > 1 and have != want:
                                issues.append(f'{tag}: header stream offsets {have}, actual {want}')
                        if guid in bundle_chunks:
                            x, bf = bundle_chunks[guid]
                            start = sum(sizes[:first])
                            if x.logical_offset != start or x.logical_size != total - start:
                                issues.append(f'{tag}: bundle chunk covers {x.logical_offset}+{x.logical_size}, '
                                              f'first mip {first} needs {start}+{total - start}')
                            data = fb.decode_cas(g.read(bf, root) if bf.patch else g.read(bf), g.root)
                            if len(data) != x.logical_size:
                                issues.append(f'{tag}: bundle chunk decodes to {len(data)} bytes, manifest says {x.logical_size}')
                    except Exception as e:
                        issues.append(f'{name}: could not check ({e})')
    return count, issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('mods', nargs='*')
    a = p.parse_args()
    g = fb.GameData(a.game)
    mods_dir = os.path.join(a.game, 'Mods')
    enabled = {}
    try:
        for row in json.load(open(os.path.join(mods_dir, 'mods.json')))['mods']:
            enabled[row['name']] = row['enabled']
    except Exception:
        pass
    names = a.mods or sorted(d for d in os.listdir(mods_dir) if not d.startswith('.') and os.path.isdir(os.path.join(mods_dir, d)))
    for name in names:
        count, issues = scan_mod(g, os.path.join(mods_dir, name))
        state = 'enabled' if enabled.get(name, True) else 'disabled'
        print(f'{name} [{state}]: {count} textures, {len(issues)} problem(s)')
        for i in issues[:12]:
            print('    ' + i)
        if len(issues) > 12:
            print(f'    ... and {len(issues) - 12} more')


if __name__ == '__main__':
    main()
