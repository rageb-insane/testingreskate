"""One-off: copies the asset set of a working custom-shoe mod (KP_Creates-Jordan4_Blackcats)
into template_shoe/, for make_shoe_mod.py to clone. Only its structure is reused; the
mesh, textures, names and ids are all replaced when a shoe is built."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402

GAME = sys.argv[1] if len(sys.argv) > 1 else r'E:\Steam\steamapps\common\Skate'
MOD = 'KP_Creates-Jordan4_Blackcats'
OUT = os.path.join(HERE, 'template_shoe')
MESH_ID = 'b858a4f9-8fc1-4489-b430-079e6cb60cbc'
SET_ID = '439c697c-ecab-41da-94c2-ab46a8137617'
ITEM_PRESET_ID = 'b469c131-b0c7-4a83-967f-5f1846d53d35'

WANT = {
    'items/cust_shoes/own_jordan4_blackcats': 'item',
    'thumbnail/tool/own_jordan4_blackcats': 'thumb_tool',
    'thumbnail/tool/own_jordan4_blackcats_lrg': 'thumb_tool_lrg',
    'thumbnail/cdn/img_own_jordan4_blackcats': 'thumb_cdn',
    'thumbnail/cdn/img_own_jordan4_blackcats_lrg': 'thumb_cdn_lrg',
    f'characters/maincharacters/reskate/{MESH_ID}_geometry': 'geometry',
    f'characters/maincharacters/reskate/{MESH_ID}_mesh': 'mesh',
    f'characters/maincharacters/reskate/{MESH_ID}_variation_2': 'variation',
    f'characters/customization/reskate/textureedits/{SET_ID}_ap': 'preset_set',
    f'characters/customization/reskate/textureedits/{ITEM_PRESET_ID}_ap': 'preset_item',
}
for key in ('89aaa101-e0f3-4084-30fc-c514010073d2', '0bb23445-4861-79a3-30fc-c5140100e899',
            '3f4cd4ca-ab37-af17-30fc-c5140100f7d6', '00c854f4-547f-9695-30fc-c5140100edb5'):
    WANT[f'characters/customization/reskate/textureedits/{SET_ID}_ap_{key}'] = f'tex_set_{key[:8]}'
WANT[f'characters/customization/reskate/textureedits/{ITEM_PRESET_ID}_ap_0bb23445-4861-79a3-30fc-c5140100e899'] = 'tex_item_0bb23445'


def main():
    g = fb.GameData(GAME)
    root = os.path.join(GAME, 'Mods', MOD)
    _, bundles, _ = fb.read_toc(open(os.path.join(root, 'Win32', 'items.toc'), 'rb').read())
    os.makedirs(OUT, exist_ok=True)
    info = {'source_mod': MOD, 'mesh_id': MESH_ID, 'set_id': SET_ID, 'item_preset_id': ITEM_PRESET_ID,
            'assets': {}, 'bundles': {}}
    for b in bundles:
        files, _ = fb.read_bundle_region(b.region)
        ebx, res, ch, meta = g.manifest(files, root)
        tree, _ = fb.read_db(meta, 0)
        added = []
        for i, a in enumerate(ebx + res + ch):
            f = files[1 + i]
            if not f.patch:
                continue
            added.append(f'{a.kind}:{a.name}')
            label = WANT.get(a.name)
            if a.kind == 'chunk':
                j = i - len(ebx) - len(res)
                info['assets'].setdefault('chunk_meta', {})[a.name] = fb.write_db(tree['children'][j]).hex()
                continue
            if not label:
                continue
            ext = 'ebx' if a.kind == 'ebx' else 'res'
            with open(os.path.join(OUT, f'{label}.{ext}'), 'wb') as h:
                h.write(g.payload(f, root))
            rec = {'name': a.name, 'kind': a.kind}
            if a.kind == 'res':
                rec.update(res_type=a.res_type, res_meta=a.res_meta.hex())
            info['assets'][f'{label}.{ext}'] = rec
        info['bundles'][b.name] = added
    missing = [v for v in WANT.values() if not any(k.startswith(v + '.') for k in info['assets'])]
    if missing:
        raise SystemExit(f'template assets missing: {missing}')
    json.dump(info, open(os.path.join(OUT, 'template.json'), 'w'), indent=1)
    print('captured', len(info['assets']), 'assets to', OUT)


if __name__ == '__main__':
    main()
