from pathlib import Path
from typing import Iterable
from itertools import accumulate, chain, filterfalse
import unicodedata
import argparse
from contextlib import contextmanager
from io import BytesIO
import subprocess
import json
import sys
from collections import defaultdict

from pdfminer.psparser import PSStackParser, PSLiteral, KWD, literal_name
from pdfminer.psexceptions import PSEOF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTCurve
from pdfminer.pdffont import PDFType1Font, PDFTrueTypeFont
from pdfminer.pdftypes import resolve1

import pymupdf
import numpy as np
import OpenEXR
import freetype

from encode_curve import encode_curve

def to_nfkd(ch):
    return unicodedata.normalize('NFKD', ch)

# https://superuser.com/questions/669130/double-latin-letters-in-unicode-_ligatures
def is_ligature(ch):
    s = "".join(filterfalse(unicodedata.combining, to_nfkd(ch)))
    return 1 < len(s) and s.isascii() and s.isalpha()

LIGATURES = {to_nfkd(ch): ch for ch in map(chr, range(0, 0x10FFFF + 1)) if is_ligature(ch)}

VERTICAL_PRESENTATION_FORMS = \
    {to_nfkd(ch): ch for ch in map(chr, chain(range(0xFE10, 0xFE20), range(0xFE30, 0xFE45), (0xFE47, 0xFE48)))}

def try_recover_unicode(o):
    ch = o.get_text()
    d = to_nfkd(ch)
    if d[:5] == '(cid:':
        raise KeyError
    if d in LIGATURES:
        return LIGATURES[d]
    if o.font.is_vertical() and d in VERTICAL_PRESENTATION_FORMS:
        return VERTICAL_PRESENTATION_FORMS[d]
    return ch

class MyType1HeaderParser(PSStackParser):
    KEYWORD_PUT = KWD(b"put")
    def do_keyword(self, pos, token):
        if token is self.KEYWORD_PUT:
            ((_, key), (_, value)) = self.pop(2)
            if isinstance(key, int) and isinstance(value, PSLiteral) and literal_name(value) != '.notdef':
                self.add_results((key, literal_name(value)))

def get_type1_cid2gid(font):
    r = []
    if "Encoding" in font.spec:
        cid = 0
        for x in resolve1(font.spec["Encoding"])["Differences"]:
            if isinstance(x, int):
                cid = x
            else:
                r.append((cid, literal_name(x)))
                cid += 1
    elif hasattr(font, "fontfile"):
        data = font.fontfile.get_data()[:font.fontfile["Length1"]]
        parser = MyType1HeaderParser(BytesIO(data))
        try:
            while True:
                r.append(parser.nextobject())
        except PSEOF:
            pass
    else:
        print(f"Cannot determine encoding for {font.fontname}! Try using --unicode {font.fontname}")
        return {}
    face = freetype.Face(f"fonts\\{font.fontname}.pfa")
    return {cid: face.get_name_index(bytes(cname, "ascii")) for cid, cname in r}

class FontIndexer:
    def __init__(self, font, by_unicode):
        self.unidentified_cids = set()
        # Certainly wrong but kind of works?...
        if by_unicode:
            self._indexer_map = try_recover_unicode
        elif isinstance(font, PDFType1Font) and not isinstance(font, PDFTrueTypeFont):
            self.cid2gid = get_type1_cid2gid(font)
            self._indexer_map = lambda o: self.cid2gid[o.cid]
        else:
            self._indexer_map = lambda o: o.cid
        
    def indexer_map(self, o):
        try:
            return self._indexer_map(o)
        except KeyError:
            self.unidentified_cids.add(o.cid)
            return None
        

class GlyphIndexer:
    def __init__(self, glyphs, by_unicode_fonts):
        self.font_indexers = {}
        self.by_unicode_fonts = by_unicode_fonts
        glyphsets = defaultdict(set)
        for o in glyphs:
            if to_nfkd(o.get_text()) == ' ':
                continue
            if o.font not in self.font_indexers:
                self.font_indexers[o.font] = FontIndexer(o.font, o.fontname in by_unicode_fonts)
            if (g := self.indexer_map(o)) is not None:
                glyphsets[o.fontname].add(g)
        self.glyphs_sorted = {k: sorted(v) for k,v in glyphsets.items()}
        self.atlas_index = {x: i for i,x in enumerate((k,g) for k,v in self.glyphs_sorted.items() for g in v)} 

    def fontnames(self):
        return self.glyphs_sorted.keys()

    def by_unicode(self, fontname):
        return fontname in self.by_unicode_fonts

    def indexer_map(self, o):
        return self.font_indexers[o.font].indexer_map(o)

    def get_atlas_id(self, o):
        if to_nfkd(o.get_text()) == ' ':
            return None
        return self.atlas_index.get((o.fontname, self.indexer_map(o)))

    def write_glyphset_files(self):
        for k,v in self.glyphs_sorted.items():
            if self.by_unicode(k):
                charset = '"{}"'.format("".join( "\\"+c if c in "\\\"" else c for c in v ))
                Path(f"fonts\\{k}-unicode.txt").write_text(charset, encoding="utf-8")
            else:
                Path(f"fonts\\{k}.txt").write_text(",".join(map(str, v)))
    
    def unidentified_cids(self):
        return [(font.fontname, list(indexer.unidentified_cids)) for font, indexer in self.font_indexers.items() 
                if indexer.unidentified_cids]

@contextmanager
def pymupdf_ctx(filename):
    file = pymupdf.open(filename)
    try:
        yield file
    finally:
        file.close()

def extract_embedded_fonts(filename):
    with pymupdf_ctx(filename) as doc:
        # Ignores duplicate embedded fonts created by LaTeX
        fonts2xref = {fontname: xref for i in range(doc.page_count) 
                      for xref, _, _, fontname, _, _ in doc.get_page_fonts(i)}
        
        font_filepaths = {}
        for xref in fonts2xref.values():
            fontname, ext, _, buffer = doc.extract_font(xref)
            if ext == "n/a" or not buffer:
                continue
            out_file = f"fonts\\{fontname}.{ext}"
            Path(out_file).write_bytes(buffer)
            font_filepaths[fontname] = out_file
        return font_filepaths
        
def tree_walk(o):
    yield o
    if isinstance(o, Iterable):
        for i in o:
            yield from tree_walk(i)

def is_uniformly_scaled(o):
    return o.matrix[0] == o.matrix[3] and o.matrix[1] == 0 and o.matrix[2] == 0

def get_page_data(page, glyph_indexer):
    half_width = page.width / 2
    half_height = page.height / 2
    scale = 1 / max(page.width, page.height)
    def rescale(pt):
        x, y = pt
        return ((x - half_width) * scale, (y - half_height) * scale)
    rescale.scale = scale

    chars = []
    others = []
    for o in tree_walk(page):
        if isinstance(o, LTChar):
            atlas_id = glyph_indexer.get_atlas_id(o)
            if atlas_id is None:
                continue
            if is_uniformly_scaled(o):
                chars.append((*rescale(o.matrix[4:]), o.size * scale, atlas_id))
            else:
                others.extend([np.array(o.matrix[:4]) * o.fontsize * scale,
                               (*rescale(o.matrix[4:]), atlas_id, 0)])
        elif isinstance(o, LTCurve):
            others.extend(encode_curve(o, rescale=rescale))
    meta = (len(chars) + len(others), len(chars), len(others) // 2, 0)
    return meta, chars + others

def min_pot_rect_greater(n):
    log2size = (n - 1).bit_length()
    return 1 << (log2size // 2), 1 << (log2size - log2size // 2)

def page_offset_float32(offset, m):
    return offset, *m[1:]

def page_offset_float16(offset, m):
    high, low = offset >> 14, offset & 0x3FFF # avoid NaNs
    buf_uint16 = np.array((high, *m[1:3], low), dtype=np.uint16)
    return np.frombuffer(buf_uint16, dtype=np.float16)

def encode_pdf_tex(pages, glyph_indexer, atlas_bounds, dtype):
    data = [get_page_data(p, glyph_indexer) for p in pages]
    meta = [d[0] for d in data]
    max_num_quads = max(sum(m[1:]) for m in meta)
    print(f"Page Size: {pages[0].width}, {pages[0].height}")
    print(f'# of pages: {len(data)}')
    print(f'# of triangles required: {(max_num_quads - 1) // 32 + 1}')
    if atlas_bounds:
        print(f'Atlas Offset: {len(atlas_bounds)}')
    offsets = accumulate((m[0] for m in meta), initial = len(data) + len(atlas_bounds))
    page_offset_f = page_offset_float16 if dtype == np.float16 else page_offset_float32
    final = list(chain(atlas_bounds,
                       map(page_offset_f, offsets, meta),
                       (x for d in data for x in d[1])))

    height, width = min_pot_rect_greater(len(final))
    tex = np.array(final, dtype=dtype)
    tex.resize(height, width, 4)
    tex = tex[::-1]
    return np.ascontiguousarray(tex)

def write_exr_texture(tex, filename):
    channels = { "RGBA" : tex }
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
               "type" : OpenEXR.scanlineimage }
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(filename)
    print(f"Encoded pdf in {filename}.")

def run_atlas_gen(outname, indexer, font_filepaths, packed, atlas_size, do_not_regen_atlas):
    if not indexer.fontnames():
        print("pdf does not contain any glyphs. Skipping atlas generation")
        return []

    font_args = []
    all_fontfiles_available = True
    for fontname in indexer.fontnames():
        if fontname in font_filepaths:
            fontfile = font_filepaths[fontname]
        else:
            for ext in ("otf", "ttf"):
                fontfile = f"fonts\\{fontname}.{ext}"
                if Path(fontfile).is_file():
                    print(f"{fontname} is not embedded. Using {fontfile}")
                    break
            else:
                all_fontfiles_available = False
                print(f"{fontname} is not embedded and no font file with matching name was found.")

        args = ["-font", fontfile] + (["-charset", f"fonts\\{fontname}-unicode.txt"] if indexer.by_unicode(fontname) else 
                                      ["-glyphset", f"fonts\\{fontname}.txt"])
        if font_args:
            font_args.append("-and")
        font_args.extend(args)
    
    _, cols = min_pot_rect_greater(len(indexer.atlas_index))
    dim_args = ["-dimensions", str(atlas_size), str(atlas_size)] if atlas_size else ["-pots"]
    grid_args = [] if packed else ["-uniformgrid", "-uniformorigin", "on", "-uniformcellconstraint", "pots", "-uniformcols", f"{cols}"]
    all_args = ["msdf-atlas-gen.exe", "-imageout", f"out\\{outname}_atlas.png", "-json", f"out\\{outname}_atlas.json", 
                "-outerpxpadding", "2"] + dim_args + grid_args + font_args

    if all_fontfiles_available:
        if not do_not_regen_atlas:
            print("=================================== Running MSDF Atlas Gen ===================================")
            # print(" ".join(all_args))
            subprocess.run(all_args)
            print("==============================================================================================")
        
        atlas_json = json.loads(Path(f"out\\{outname}_atlas.json").read_text())
        atlas_size = atlas_json["atlas"]["width"]
        bounds = (bounds
                  for variant in atlas_json.get("variants", (atlas_json,))
                  for glyph in variant["glyphs"]
                  for bounds in (np.array(list(glyph["planeBounds"].values())), 
                                 np.array(list(glyph["atlasBounds"].values())) / atlas_size))
        if packed:
            return list(bounds)
        else:
            print(f"Plane Bounds: {next(bounds)}")
    else:
        if packed:
            sys.exit("Cannot encode pdf without generating atlas first when --packed is used! Exiting...")
        else:
            print("Please acquire missing fonts then run the following command to generate the glyph atlas:")
            print(" ".join(all_args))

    return []


class Everything(set):
    def __contains__(self, item):
        return True
EVERYTHING = Everything()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument('--unicode', help="extract a list of fonts by unicode. try this when a font has unidentified cids", nargs="*", metavar="FONT")
    parser.add_argument("--unicode-all", help="extract all fonts by unicode", action="store_true")
    parser.add_argument("--packed", help="generate a tightly packed atlas. recommended for math papers/books since these tend to have large variance between glyph sizes, and a large amount of atlas space will be otherwise wasted", action="store_true")
    parser.add_argument("--atlas-size", help="specify atlas size. if the default size is not large enough. PLEASE USE A POWER OF TWO LIKE 1024, 2048, 4096", type=int)
    parser.add_argument("--half", help="encode pdf as RGBA Half, which reduces texture memory usage", action="store_true")
    parser.add_argument("--do-not-regen-atlas", help="for testing purpose only", action="store_true")

    args = parser.parse_args()
    pages = list(extract_pages(Path(args.filename)))
    outname = args.filename[:-4] + ("_packed" if args.packed else "") + ("_half" if args.half else "")
    by_unicode_fonts = EVERYTHING if args.unicode_all else set(args.unicode) if args.unicode else set()

    Path.mkdir(Path("fonts"), exist_ok=True)
    Path.mkdir(Path("out"), exist_ok=True)
    font_filepaths = extract_embedded_fonts(args.filename)

    glyphs = filter(lambda o: isinstance(o, LTChar), tree_walk(pages))
    indexer = GlyphIndexer(glyphs, by_unicode_fonts)
    if indexer.unidentified_cids():
        print("Failed to identify the following cids:")
        for fontname, cids in indexer.unidentified_cids():
            print(f"{fontname}: {cids}")

    indexer.write_glyphset_files()

    atlas_bounds = run_atlas_gen(outname, indexer, font_filepaths, args.packed, args.atlas_size, args.do_not_regen_atlas)

    dtype = np.float16 if args.half else np.float32
    tex = encode_pdf_tex(pages, indexer, atlas_bounds, dtype)
    write_exr_texture(tex, f"out\\{outname}.exr")


