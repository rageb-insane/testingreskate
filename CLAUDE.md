# ReSkate Deck: skate. mod tools

Python tools that build cosmetic mods for **skate.** (EA 2025, Frostbite engine). They're loaded through the
ReSkate mod loader (github.com/Dingo-Shenanigans/ReSkate). The tools make decks, grip tape, crop-top logos,
recoloured skinny jeans, custom 3D shoes, a pants-tuck mod and modpacks. The mod author name is
**socioculture**: mod folders are `socioculture-<Package>`, and builders default `--author` to it.

## Where things run

- The game lives at `E:\Steam\steamapps\common\Skate` (`DEFAULT_GAME` in `tool/make_deck_mod.py`), on the
  user's Windows PC. Mods go in its `Mods\` folder, and ReSkate writes its merged output to
  `Mods\.reskate\Win32\items.toc`.
- **A cloud session has no game files.** Every builder and verifier reads the game's TOCs, bundles and CAS
  archives. In the cloud you can read and edit the code and reason about it, but you can't build, verify,
  render from game meshes or install. Say so rather than pretending a build ran, and leave builds and
  in-game checks for a local session.
- The local setup is Python 3.14 with numpy, Pillow, opencv-python and fast-simplification. There's no
  scipy. `tool/ebxtool/ebxtool.exe` is built from `ebxtool.cpp` plus the GPL ReSkate sources in
  `tool/ReSkate` by `tool/ebxtool/build.bat`, which needs MSVC.
- Not in the repo, and never to be added: `models/` (licensed 3D models) and most of `mods/` (built
  output). Models from CGTrader and Zertux are **personal use only, do not redistribute**.
- The GitHub repo is `rageb-insane/testingreskate` (public). It holds the tools, the `Make *.bat`
  launchers, the deck and grip templates and this file, laid out as the desktop folder is, so the
  builders' defaults hold (`--out` is `mods/`, the Yeezy colourway is `models/nike_air_yeezy/`). The
  2026-10-06 web upload put the Yeezy GLB, `red_october.json` and two built mods on `main` against the
  rule above; a cloud session moved them into `models/` and `mods/`, and whether they stay is the
  user's call. The web uploader skips dotfiles, so those mod folders lack `.reskate-studio-patch`
  (a rebuild writes it again) and their `-tuck.json` files.

## Standing rules from the user (follow them always)

- **Never install while skate. is running.** Run `tasklist /FI "IMAGENAME eq Skate.exe"` yourself before
  any install or `--install`. Build into `mods/`, and only install when the user asks. Don't auto-start
  install watchers.
- Only tell the user to copy mods once **every** build in a set has finished. They once copied
  half-built folders and the install was left inconsistent. The pants tuck mod and the shoes whose item
  it patches must be copied together.
- **New shoe models, "first rule":** the first build only scales and places the model (`--fit scale`).
  Adjust nothing until the user has seen it in game. Report what the measurements predict, but don't
  fix it yet.
- **Don't deform models** ("without fucking the model up"). Skinning, culling and texture changes are
  fine; reshaping is not. The user saw Dunk-High reshaping warp a shoe and rejected it.
- Don't change mods the user has confirmed in game as a side effect of builder changes. New behaviour
  is opt-in, and the old behaviour stays the default.
- **Deleted things stay deleted.** The user deleted the Dior Jordan 1 and the old Sketchfab Chicago. Never
  rebuild, copy back, register or patch them, even for compatibility. The only shoes are the CGTrader
  Air Jordan 1 Chicago, the Air Jordan 4 Pure Money, and the Nike Air Yeezy 2 (Black Solar Red and Red
  October).
- Don't build, rebuild or fix the "Naker" deck (any name, e.g. `Black_Naker_Red_Deck`); Claude declined it.
  Don't unlock paid items.
- Write patches as script files, not long shell heredocs; quoting broke repeatedly on Windows.

## Layout

- `Make *.bat`: drag-and-drop launchers for the user.
- `tool/fb.py`: Frostbite I/O for TOCs, bundles, CAS, manifests and chunk metadata (`GameData`).
- `tool/meshset.py`: MeshSet reader and `decode_section`. `tool/tangentspace.py` holds the TangentSpace
  codec.
- `tool/make_deck_mod.py`: decks, plus shared helpers (`CasWriter`, `edit_ebx`, `ebx_info`, `djb`,
  `new_guid`). It's verified by `verify_mod.py`.
- `tool/make_grip_mod.py`, `make_top_mod.py`, `make_jeans_mod.py` and `make_modpack.py` each have a
  `verify_*.py`.
- Shoes:
  - `make_shoe_mod.py` is the main builder. `shoe_prep.py` stands the model up, picks the foot and makes
    the mirrored pair. `shoe_mesh.py` builds the mesh: fit, skinning, collar rims.
  - `meshreduce.py` reduces the mesh. `gltf.py` and `objfile.py` are the loaders. `atlas.py` pads UVs.
  - `verify_shoe_mod.py` verifies a shoe. `template_shoe/` holds the captured game assets the builder
    clones, made by `capture_shoe_template.py`.
- Pants tuck: `make_tuck_mod.py` (pants and body culling data), `tuck_items.py` (patches each shoe's
  item CulledRegions in place) and `body_region_groups.json` (a cache).

## Current shoes (run from `tool/`)

The Yeezy builds, Black first, then Red October with
`--colours ../models/nike_air_yeezy/red_october.json` and `--name "Nike Air Yeezy 2 Red October"`:
```
python make_shoe_mod.py "../models/nike_air_yeezy/nikeyeezyblack.glb" --name "Nike Air Yeezy 2 Black Solar Red" --fit scale --base vertclassic --tuck --gold aglet --unmirror Plane --credit "Model: Nike Air Yeezy by Zertux - personal use only, do not redistribute"
```

For the collar fix (open item 2) add `--tuck-rim lip` to both Yeezy builds and to the AJ4 build below;
`--tuck-margin 3` adds 3 mm of room above the lip at the heel if the jeans still cut it. Nothing has
been rebuilt with these yet.

The AJ4 Pure Money build, with `M=../models/air_jordan_4_jumpman_variety_pack` and
`T=$M/Jordan4_VarietyPack_Textures/PureMoney`:
```
python make_shoe_mod.py "$M/Jordan4_LowPoly.obj" --name "Air Jordan 4 Pure Money" --texture "$T/J4_PureMoneyOneMesh_BaseColor.png" --normal "$T/J4_PureMoneyOneMesh_Normal.png" --normal-style directx --texture-size 2048 --pad-seams --fit scale --base vertclassic --tuck --tint 0.4,0.4,0.4 --fill-holes 0.86 --metallic "$T/J4_PureMoneyOneMesh_Metallic.png" --credit "Model: Air Jordan 4 Jumpman Variety Pack (CGTrader) - personal use only, do not redistribute"
```

**AJ1 Chicago** (CGTrader pack, `Jordan1_Base_LowPoly.obj`, Chicago texture, DirectX normal map) uses
`--fit model` on Dunk High bones with `skin_above_rim`. The user calls it "pretty much perfect", and the
installed build was restored from an earlier good build. Don't rebuild it unless asked. The Dunk High
bones were "just for the jordan 1": other models use `--base vertclassic` (or puffy90s).

A shoe build with `--tuck` reruns `make_tuck_mod` itself and patches every registered shoe's item.
Afterwards, copy all of these together: the two Yeezys, `socioculture-High_Top_Pants_Tuck`, the Chicago
and the AJ4.

## Format facts and lessons (hard-won; trust these)

**Hashes and ids**
- HashedAssetKey/NameHash use djb2-xor 32 (5381, `*33 ^ c`). The bundle-ref table uses FNV-1a64 on the
  lowercased name. A manifest sha1 is the sha1 of the CAS-encoded bytes.
- Resource ids must have bit 0 set.
- A texture RES header's bytes 0-7 (mip offsets) and 120-127 (djb64 of the name) must never be copied
  from a template.

**Textures**
- Builders use the community "whole chunk in bundle" layout (header byte 31 = 0, firstMip 0). The game's
  streamed layout gave random noise or crashes.
- When two mods ship the same bundle, ReSkate keeps only the first one's chunk metadata. So each custom
  shoe gets a vertclassic colorway slot (00001-00010) no other mod ships (`pick_colorway`).

**Shoe mesh**
- One section; 16-bit indices, so both feet together hold at most 65,535 vertices (`MAX_VERTICES` is
  64000).
- Half-float positions (format 8), 2×4 bone influences, UByte4N weights; the TangentSpace is the axis-angle
  variant.
- Shoe texture slots: 0bb23445 base_c, 3f4cd4ca base_ny, 00c854f4 msk, 89aaa101 stitches. There's no
  metal input, so chrome and gold are baked into the colour.
- Game `_ny` normal maps are OpenGL-style, and our encoder expects DirectX. The CGTrader packs are DirectX.
- `shoe_prep.stand_up` finds the sole as the most flat area at one extreme with little opposite. On the
  Yeezy 2 that is marginal: the big flat side panel scores nearly as high as the patterned outsole, the
  GLB passes, the OBJ (same shape) falls the other way and lies on its side. `--up y` (or any axis) says
  the model already stands and skips the search; the search itself is unchanged for every other model.
- The game is right-handed: y up, toes at +z, and **x>0 is the LEFT foot**. The game culls back faces.
  Inside-out triangles in a model show as missing pieces in game; `_match_twin_winding` fixes them, gated
  by TWIN_GAP.

**Region culling (how pants tuck into shoes)**
- A `*_complex_dmpreset` (DingoMorphPreset) has `RegionCullingData`. The format is written out in the
  `make_tuck_mod.py` docstring.
- Items' CulledRegions hide matching triangles. The game itself only uses this on the body; our tuck mod
  adds it to the 19 single-section pants.
- The body has exactly **32 regions**, which looks like a hard bitmask limit: regions 33 and up are never
  hidden. Foot is 0x7C7EB057 (4.2-13.3 cm) and shin 0x7c83f1dd.
- New body regions must reuse slots freed by merging regions the game always hides together ([3,4] and
  [20,22]).
- Each tucking shoe writes `mods/<mod>-tuck.json` with its own collar rim per 5° sector. Pants regions are
  the overlaps of the shoes' sets. A triangle is hidden only if it's **wholly** under the rim.

**Pants**
- Pants positions are Float3, with no cut-line data.
- Pants morph with the character's body sliders (calf and feet regions in the DingoMorph graph); shoes
  don't. Rest-pose checks can miss in-game overlap.

**Untextured models**
- These get a palette texture (`gltf.colour_palette`). Colourways come from `--colours` JSON, gold aglets
  from `--gold`.
- An OBJ without textures goes through the same palette: `objfile.load` lists its materials glTF-style
  (the .mtl's `Kd` taken as linear, light grey without a .mtl) with each primitive's material an index
  into them, so `--colours`, `--gold` and `--unmirror` match the .mtl's material names. An OBJ whose
  .mtl names textures still uses them (a missing one for the main material now stops the build with a
  message instead of crashing later).

**Mesh reduction (`meshreduce`)**
- `sink_covered` comes first (1.5 mm). Studs are rebuilt per piece, and palette parts under 2,000 vertices
  are kept as modelled.
- `unstable_triangles` drops slivers that half floats would flip.
- When the pair is one shoe mirrored, reduce once and mirror the result.

**Rejected approaches (don't retry them)**
- Inflating surfaces.
- Deleting never-seen surfaces by visibility.
- Auto-flipping a mirrored sole texture to un-mirror its lettering.
- The smooth collar fit (`--collar smooth`) as a default.

**Preview renders**
- With screen-right = `cross(view_dir, up)`, `cross(up, view_dir)` mirrors the image. An old scratch
  renderer did this, and text checks made with it read the wrong way.
- Render with back-face culling like the game. Two-sided previews hid inside-out pieces.

## Open work (Yeezy, paused 2026-10-06; nothing rebuilt since)

1. **Strap NIKE reads backwards on both feet** (user screenshots; "not centered and in line").
   - Every node in `nikeyeezyblack.glb` has scale (-1,-1,-1), so the loaded model is mirrored: the
     model's own foot (`twin[0]`, the right foot) has backwards lettering, and the mirrored copy reads
     correctly.
   - `--unmirror` currently flips `twin[1]`, which is wrong. Flip the foot whose source glTF nodes have a
     negative determinant (`gltf.load` knows the world matrix).
   - Reflect each text part across its in-plane minimum-area-rectangle long axis, through the rectangle
     centre, **not** the PCA axis. On the italic NIKE the PCA axis is about 9° off, so the flipped text
     turned 20° and moved 2.6 mm off the strap tab's centre. In the model the text is centred on the tab
     (0.4 mm, -2.2°).
   - The text parts are palette materials containing "Plane": `24.12-_Plane.001` on the strap and
     `30.15-_Plane.001` on the outsole.
   - The user also has the model as Zertux's original OBJ (`model.obj`, a Blender 2.49 export, 41 MB, no
     .mtl, the same 636,214 triangles as the GLB). It is **not in the repo** (public; personal use only)
     and belongs beside the GLB in `models/nike_air_yeezy/`. It loads through `objfile` in 5 s with 54
     materials named like `8.4-_lace_`, `6.3-_rubber_`, `4.2-_sole_`, `26.13-_tongue_in`, so a colourway
     can colour laces, rubber and the tongue on their own (in the GLB those share other materials);
     `--gold aglet` and `--unmirror Plane` match the same parts as on the GLB. The GLB's nodes carry
     scale (-1,-1,-1), so the GLB is the OBJ mirrored, confirmed in the cloud: stood up, the OBJ's own
     foot is the **left** shoe with the same box as the GLB's right (7.86 x 4.60 x 3.27 model units), the
     GLB's own foot the right. So a build from the OBJ puts the readable lettering on the model's own
     foot and `--unmirror` flips the mirrored one, which is what the code already does; check that in
     game before touching `--unmirror` for the GLB. Two things the OBJ needs: `--up y` (`stand_up`'s sole
     search picks its flat side panel, see below) and the faster `shoe_prep._objects` (it makes 19,594
     loose pieces against the GLB's 3,198; the old pair-by-pair loop took half an hour on it, the
     vectorised one 10 s with identical groups). Untested beyond `shoe_prep.prepare`: the GLB's command
     with the model swapped and `--up y` added is the one to try, first as `--fit scale` only.
2. **Jeans overlap the back of the collar** in game (skinny jeans). Code done 2026-10-06 in a cloud
   session (no game files there), nothing rebuilt yet; the AJ4 is built the same way and has the same issue.
   - Cause: `shoe_mesh.own_collar_rims` lowered each 5° sector to the lowest of itself and its neighbours.
     At the back that gave a rim of 13.2 cm where the collar lip's real top is about 14.0 cm. The jeans
     there sit about 4 mm outside the lip and were kept down to about 13.0 cm, so they covered it; at
     210° they passed through it.
   - Fix, opt-in (the old measure stays the default, per the standing rule): `--tuck-rim lip` takes each
     sector's own highest point, the lip's real top (`own_collar_rims(lip=True)`; a sector no point falls
     in borrows its lower neighbour). `--tuck-margin MM` hides pants up to MM above the lip at the heel,
     easing to nothing at the toe, so no gap opens over the tongue. Both go into `<mod>-tuck.json`, so
     the tuck mod, the body foot split and `skin_collar_to_leg` all follow the same rim. A lip build logs
     how far the lip reads above the sector-min rim, and the lip heights over 150-210°.
   - `--hide-feet auto` uses the same rim. If the Yeezy's lowest lip point is at or above 13.3 cm, a lip
     build hides the body's feet where the old build kept them drawn (the log says which); `--hide-feet
     no` keeps the old choice.
   - Left to do locally: rebuild both Yeezys and the AJ4 with `--tuck-rim lip` (each reruns the tuck),
     check vertical cross-sections through the back of the collar against the skinny jeans, confirm in
     game, then copy the Yeezys, the AJ4 and `socioculture-High_Top_Pants_Tuck` together.
   - Already done earlier: the collar's top 3 cm is skinned like the leg (`skin_collar_to_leg`).
3. Minor: light streaks on the Yeezy toe box (the glow sole under the suede).
