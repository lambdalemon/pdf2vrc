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
import glob
from collections import defaultdict

from pdfminer.psexceptions import PSEOF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTCurve, LTImage, LTFigure
from pdfminer.utils import mult_matrix
from pdfminer.image import ImageWriter
from pdfminer.pdffont import PDFType1Font, PDFTrueTypeFont, PDFCIDFont

import pymupdf
import numpy as np
import OpenEXR
import freetype
from PIL import Image
import PyTexturePacker

from encode_curve import encode_curve, to_rgba

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
        return 0
    if d in LIGATURES:
        return LIGATURES[d]
    if o.font.is_vertical() and d in VERTICAL_PRESENTATION_FORMS:
        return VERTICAL_PRESENTATION_FORMS[d]
    return ch

def get_unicode2gid(fontfile):
    face = freetype.Face(fontfile)
    return lambda o: face.get_char_index(try_recover_unicode(o))

def get_type1_cid2gid(font, fontfile):
    face = freetype.Face(fontfile)
    cid2gid = {cid: face.get_name_index(bytes(cname, "ascii")) for cid, cname in font.encoding.items()}
    return lambda o: cid2gid.get(o.cid)

class FontIndexer:
    def __init__(self, font, fontfile, by_unicode):
        self.unidentified_cids = set()
        # This will do for now...
        if fontfile is None:
            self._indexer_map = lambda o: None
        elif by_unicode:
            self._indexer_map = get_unicode2gid(fontfile)
        elif isinstance(font, PDFType1Font):
            self._indexer_map = get_type1_cid2gid(font, fontfile)
        elif isinstance(font, PDFCIDFont) and font.cidcoding == "Adobe-Identity":
            self._indexer_map = lambda o: o.cid
        else:
            self._indexer_map = get_unicode2gid(fontfile)

    def indexer_map(self, o):
        if g := self._indexer_map(o):
            return g
        else:
            self.unidentified_cids.add(o.cid)
            return None

def locate_external_fontfile(fontname):
    for ext in ("otf", "ttf"):
        fontfile = f"fonts\\{fontname}.{ext}"
        if Path(fontfile).is_file():
            print(f"{fontname} is not embedded. Using {fontfile}")
            return fontfile
    else:
        print(f"{fontname} is not embedded and no font file with matching name was found.")
        return None

class GlyphIndexer:
    def __init__(self, glyphs, font_filepaths, by_unicode_fonts):
        self.font_indexers = {}
        glyphsets = defaultdict(set)
        self.font_filepaths = font_filepaths
        for o in glyphs:
            if to_nfkd(o.get_text()) == ' ':
                continue
            if o.font not in self.font_indexers:
                if o.fontname not in font_filepaths:
                    font_filepaths[o.fontname] = locate_external_fontfile(o.fontname)
                self.font_indexers[o.font] = FontIndexer(o.font, font_filepaths[o.fontname], o.fontname in by_unicode_fonts)
            if g := self.indexer_map(o):
                glyphsets[o.fontname].add(g)
        self.glyphs_sorted = {k: sorted(v) for k,v in glyphsets.items()}
        self.atlas_index = {x: i for i,x in enumerate((k,g) for k,v in self.glyphs_sorted.items() for g in v)} 

    def fontnames(self):
        return self.glyphs_sorted.keys()

    def all_fontfiles_available(self):
        return all(f is not None for f in self.font_filepaths.values())

    def indexer_map(self, o):
        return self.font_indexers[o.font].indexer_map(o)

    def get_atlas_id(self, o):
        if to_nfkd(o.get_text()) == ' ':
            return None
        return self.atlas_index.get((o.fontname, self.indexer_map(o)))

    def write_glyphset_files(self):
        for k,v in self.glyphs_sorted.items():
            Path(f"fonts\\{k}.txt").write_text(",".join(map(str, v)))

    def unidentified_cids(self):
        return [(font, list(indexer.unidentified_cids)) for font, indexer in self.font_indexers.items()
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

def img_hash(o):
    return (str(o.stream.attrs), o.stream.get_data())

def get_page_data(page, glyph_indexer, img_index, img_id_offset, no_curve):
    half_width = page.width / 2
    half_height = page.height / 2
    scale = 1 / max(page.width, page.height)
    def rescale(pt):
        x, y = pt
        return ((x - half_width) * scale, (y - half_height) * scale)
    rescale.scale = scale

    chars = []
    others = []
    color_chars = []
    color_others = []
    last_matrix = None
    for o in tree_walk(page):
        if isinstance(o, LTChar):
            atlas_id = glyph_indexer.get_atlas_id(o)
            if atlas_id is None:
                continue
            color = to_rgba(o.graphicstate.ncolor)
            if is_uniformly_scaled(o):
                chars.append((*rescale(o.matrix[4:]), o.size * scale, atlas_id))
                color_chars.append(color)
            else:
                others.extend([np.array(o.matrix[:4]) * o.fontsize * scale,
                               (*rescale(o.matrix[4:]), atlas_id, 0)])
                color_others.extend([color, color])
        elif isinstance(o, LTCurve) and not no_curve:
            data, color = encode_curve(o, rescale=rescale)
            others.extend(data)
            color_others.extend(color)
        elif isinstance(o, LTFigure):
            last_matrix = o.matrix
        elif isinstance(o, LTImage) and (img_id := img_index.get(img_hash(o))):
            others.extend([np.array(last_matrix[:4]) * scale,
                           (*rescale(last_matrix[4:]), img_id + img_id_offset, 2048)])
            color = (0,0,0,0)
            color_others.extend([color, color])

    meta = (len(chars) + len(others), len(chars), len(others) // 2, 0)
    return meta, chars + others, color_chars + color_others

def min_pot_rect_greater(n):
    log2size = (n - 1).bit_length()
    return 1 << (log2size // 2), 1 << (log2size - log2size // 2)

def page_offset_float32(offset, m):
    return offset, *m[1:]

def page_offset_float16(offset, m):
    high, low = offset >> 14, offset & 0x3FFF # avoid NaNs
    buf_uint16 = np.array((high, *m[1:3], low), dtype=np.uint16)
    return np.frombuffer(buf_uint16, dtype=np.float16)

def encode_pdf_tex(pages, glyph_indexer, glyph_atlas_bounds, img_atlas_bounds, img_index, dtype, no_curve):
    data = [get_page_data(p, glyph_indexer, img_index, len(glyph_atlas_bounds), no_curve) for p in pages]
    meta = [d[0] for d in data]
    max_num_quads = max(sum(m[1:]) for m in meta)
    print(f"Page Size: {pages[0].width}, {pages[0].height}")
    print(f'# of pages: {len(data)}')
    print(f'# of triangles required: {(max_num_quads - 1) // 32 + 1}')
    atlas_bounds = glyph_atlas_bounds + img_atlas_bounds
    if atlas_bounds:
        print(f'Atlas Offset: {len(atlas_bounds)}')
    header_size = len(atlas_bounds) + len(pages)
    offsets = accumulate((m[0] for m in meta), initial=header_size)
    page_offset_f = page_offset_float16 if dtype == np.float16 else page_offset_float32
    final = list(chain(atlas_bounds,
                       map(page_offset_f, offsets, meta),
                       (x for d in data for x in d[1])))

    height, width = min_pot_rect_greater(len(final))
    tex = np.array(final, dtype=dtype)
    tex.resize(height, width, 4)
    tex = np.ascontiguousarray(tex[::-1])

    tex_color = np.vstack([np.zeros((header_size, 4)), [x for d in data for x in d[2]]])
    tex_color = (tex_color.astype(float) * 255).astype(np.uint8)
    tex_color.resize(height, width, 4)
    tex_color = np.ascontiguousarray(tex_color[::-1])
    return tex, tex_color

def write_exr_texture(tex, filename):
    channels = { "RGBA" : tex }
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
               "type" : OpenEXR.scanlineimage }
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(filename)
    print(f"Encoded pdf in {filename}.")

def write_color_texture(tex, filename):
    rgb, a = tex[:,:,:3], tex[:,:,3]
    is_black = np.all(rgb == 0) and np.all(np.logical_or(a == 0, a == 255))
    if not is_black:
        Image.fromarray(tex).save(filename)
        print(f"Encoded color in {filename}.")

def run_atlas_gen(outname, indexer, packed, atlas_size, do_not_regen_atlas):
    if not indexer.fontnames():
        print("pdf does not contain any glyphs. Skipping atlas generation")
        return []

    font_args = []
    for fontname in indexer.fontnames():
        args = ["-font", indexer.font_filepaths[fontname], "-glyphset", f"fonts\\{fontname}.txt"]
        if font_args:
            font_args.append("-and")
        font_args.extend(args)
    
    _, cols = min_pot_rect_greater(len(indexer.atlas_index))
    dim_args = ["-dimensions", str(atlas_size), str(atlas_size)] if atlas_size else ["-pots"]
    grid_args = [] if packed else ["-uniformgrid", "-uniformorigin", "on", "-uniformcellconstraint", "pots", "-uniformcols", f"{cols}"]
    all_args = ["msdf-atlas-gen.exe", "-imageout", f"out\\{outname}_atlas.png", "-json", f"out\\{outname}_atlas.json", 
                "-outerpxpadding", "2"] + dim_args + grid_args + font_args

    if not do_not_regen_atlas:
        print("=================================== Running MSDF Atlas Gen ===================================")
        subprocess.run(all_args)
        print("==============================================================================================")

    atlas_json = json.loads(Path(f"out\\{outname}_atlas.json").read_text())
    atlas_size = atlas_json["atlas"]["width"]
    bounds = [bounds
              for variant in atlas_json.get("variants", (atlas_json,))
              for glyph in variant["glyphs"]
              for bounds in (np.array(list(glyph["planeBounds"].values())), 
                             np.array(list(glyph["atlasBounds"].values())) / atlas_size)]
    if packed:
        return bounds
    else:
        print(f"Plane Bounds: {bounds[0]}")
        return []

def extract_images(pages, outname):
    imgs = {img_hash(o): o for o in tree_walk(pages) if isinstance(o, LTImage)}
    if not imgs:
        return [], {}
    img_dir = f".\\images\\{outname}"
    for f in glob.glob(f"{img_dir}\\*"):
        Path(f).unlink()
    img_writer = ImageWriter(img_dir)
    img_files = [img_writer.export_image(o) for o in imgs.values()]
    packer = PyTexturePacker.Packer.create(max_width=8192,
                                           max_height=8192,
                                           reduce_border_artifacts=True,
                                           enable_rotated=False,
                                           atlas_format=PyTexturePacker.Utils.ATLAS_FORMAT_JSON)
    packer.pack(img_dir, f"out\\{outname}_image_atlas")
    print(f"Packed images in out\\{outname}_image_atlas.png")
    atlas_json = json.loads(Path(f"out\\{outname}_image_atlas.json").read_text())
    atlas_w, atlas_h = atlas_json["meta"]["size"].values()
    def frame_to_atlasbounds(frame):
        x, y, w, h = frame.values()
        if w == 1:
            x += 0.5; w = 0
        if h == 1:
            y += 0.5; h = 0
        return np.array([x, atlas_h-(y+h), x+w, atlas_h-y]) / [atlas_w, atlas_h, atlas_w, atlas_h]
    bounds = [frame_to_atlasbounds(atlas_json["frames"][f]['frame']) for f in img_files]
    img_index = {s: i for i, s in enumerate(imgs)}
    return bounds, img_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--packed", help="generate a tightly packed atlas. recommended for math papers/books since these tend to have large variance between glyph sizes, and a large amount of atlas space will be otherwise wasted", action="store_true")
    parser.add_argument("--atlas-size", help="specify atlas size. if the default size is not large enough. PLEASE USE A POWER OF TWO LIKE 1024, 2048, 4096", type=int)
    parser.add_argument("--half", help="encode pdf as RGBA Half, which reduces texture memory usage", action="store_true")
    parser.add_argument('--unicode', help="extract a list of fonts by unicode", nargs="*", metavar="FONT")
    parser.add_argument("--do-not-regen-atlas", help="for testing purpose only", action="store_true")
    parser.add_argument("--no-curve", help="for testing purpose only", action="store_true")

    args = parser.parse_args()
    outname = args.filename[:-4] + ("_packed" if args.packed else "") + ("_half" if args.half else "") + ("_no-curve" if args.no_curve else "")
    by_unicode_fonts = set(args.unicode) if args.unicode else set()

    Path.mkdir(Path("fonts"), exist_ok=True)
    Path.mkdir(Path("out"), exist_ok=True)
    font_filepaths = extract_embedded_fonts(args.filename)
    pages = list(extract_pages(Path(args.filename)))
    glyphs = filter(lambda o: isinstance(o, LTChar), tree_walk(pages))
    indexer = GlyphIndexer(glyphs, font_filepaths, by_unicode_fonts)

    if not indexer.all_fontfiles_available():
        sys.exit("Please acquire missing fonts and try again.")
    if indexer.unidentified_cids():
        print("Failed to identify the following cids:")
        for font, cids in indexer.unidentified_cids():
            print(f"{font.fontname}: {cids}")
            print(font.spec)

    indexer.write_glyphset_files()

    glyph_atlas_bounds = run_atlas_gen(outname, indexer, args.packed, args.atlas_size, args.do_not_regen_atlas)
    img_atlas_bounds, img_index = extract_images(pages, outname)

    dtype = np.float16 if args.half else np.float32
    tex, tex_color = encode_pdf_tex(pages, indexer, glyph_atlas_bounds, img_atlas_bounds, img_index, dtype, args.no_curve)
    write_exr_texture(tex, f"out\\{outname}.exr")
    write_color_texture(tex_color, f"out\\{outname}_color.png")
