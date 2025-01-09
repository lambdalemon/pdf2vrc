from typing import Iterable
from itertools import product, combinations, pairwise, accumulate, cycle
import numpy as np
from numpy.linalg import norm
from pdfminer.layout import LTLine
import triangle

import faulthandler
faulthandler.enable()

def det(x0, x1):
    return np.linalg.det(np.array((x0, x1)))

def is_clockwise(x0, x1, x2):
    return det(x1 - x0, x2 - x0) < 0
    
def line_intersect(x0, d0, x1, d1):
    d = det(d0, d1)
    return np.array((-d1, d0)).T.dot(np.array((det(x0,d0), det(x1,d1)))) / d if d else None

def segment_intersect(x0, x1, x2, x3):
    d = det(x0-x1, x2-x3)
    if d == 0:
        return False
    t = np.array((det(x0-x2, x2-x3), det(x1-x0, x0-x2))) / d
    return all((0<t) & (t<1))

def cubic2quads(x0, x1, x2, x3):
    mid = (x0 + 3 * (x1 + x2) + x3) / 8
    mid_d = x3 + x2 - x1 - x0
    start_d = x1 - x0 if norm(x1-x0) > 1e-4 else x2 - x0
    end_d = x2 - x3 if norm(x2-x3) > 1e-4 else x1 - x3
    c0 = line_intersect(x0, start_d, mid, -mid_d)
    c1 = line_intersect(x3, end_d, mid, mid_d)
    return c0, mid, c1

def split_bezier(x0, x1, x2):
    c0 = (x0 + x1) / 2
    c1 = (x2 + x1) / 2
    mid = (c0 + c1) / 2    
    return c0, mid, c1

def is_black(c, cutoff=1e-2):
    if isinstance(c, Iterable):
        return len(c) == 3 and sum(c) < 3 * cutoff or len(c) == 4 and c[-1] > 1 - cutoff
    else:
        return c < cutoff

def bounding_box(verts):
    hull_vecs = [verts[x] - verts[y] for x, y in triangle.convex_hull(verts)]
    hull_vecs /= norm(hull_vecs, axis=1).reshape((-1, 1))
    rots = np.stack([np.array([d, [-d[1], d[0]]]) for d in hull_vecs])
    rotated_verts = rots.dot(verts.T)
    max_xy = np.max(rotated_verts, axis=2)
    min_xy = np.min(rotated_verts, axis=2)
    size_xy = max_xy - min_xy
    area = size_xy[:,0] * size_xy[:,1]
    best = np.argmin(area)
    center = (max_xy[best] + min_xy[best]) / 2
    halfsize = size_xy[best] / 2
    box = (center + halfsize * 1.1 * np.array([(-1, -1), (-1, 1), (1, 1), (1, -1)])).dot(rots[best])
    ext_pt = (center + halfsize * 1.05 * np.array((-1, -1))).dot(rots[best])
    return box, ext_pt

def tri_uv_packed(a,b,c):
    return -((1+a) + ((1+b)<<2) + ((1+c)<<4))

def is_dashed(dashing_style):
    return dashing_style and dashing_style[0]

def dashed_segments(dash_array, phase, length):
    phase = np.fmod(phase, sum(dash_array) * (1 + len(dash_array) % 2))
    for (x,y), is_on in zip(pairwise(accumulate(cycle(dash_array), initial=-phase)), cycle([True,False])):
        if y < 0:
            continue
        if x >= length:
            break
        start, end = max(0,x), min(length,y)
        if is_on and end - start > 1e-4:
            yield start, end

def split_until_tolerance(x, t, tolerance=0.05):
    if norm(x[2]-x[1]) + norm(x[1]-x[0]) - norm(x[2]-x[0]) < tolerance:
        return [(x,t)]
    c0, mid, c1 = split_bezier(*x)
    t_mid = (t[0] + t[1]) / 2
    return split_until_tolerance((x[0], c0, mid), (t[0], t_mid), tolerance) + \
           split_until_tolerance((mid, c1, x[2]), (t_mid, t[1]), tolerance)

# https://www.sciencedirect.com/science/article/pii/0925772195000542
def bezier_length_est(x):
    return (norm(x[2]-x[1]) + norm(x[1]-x[0]) + 2 * norm(x[2]-x[0])) / 3

def lerp(x0, x1, t):
    return (1-t) * x0 + t * x1

def bezier_eval(x,t):
    return lerp(lerp(x[0],x[1],t), lerp(x[1],x[2],t), t)

def bezier_dir_eval(x,t):
    return lerp(x[1]-x[0], x[2]-x[1], t)

def bezier_segment(x, t0, t1):
    a = bezier_eval(x,t0)
    b = bezier_eval(x,t1)
    ad = bezier_dir_eval(x,t0)
    bd = bezier_dir_eval(x,t1)
    c = line_intersect(a, ad, b, bd)
    if c is None:
        c = (a + b) / 2
    return a, c, b

def encode_line_segment(pts, linewidth, rescale):
    return [rescale(pts[0]) + rescale(pts[1]), 
            (linewidth * rescale.scale, 0, 1, 1)]

def encode_line(o, rescale):
    if not is_dashed(o.dashing_style):
        return encode_line_segment(o.pts, o.linewidth, rescale)
    else:
        data = []
        pts = np.array(o.pts)
        total_length = norm(pts[1] - pts[0])
        segment_t = np.array(list(dashed_segments(*o.dashing_style, total_length))) / total_length
        for t0, t1 in segment_t:
            dash = [lerp(pts[0], pts[1], t) for t in (t0, t1)]
            data.extend(encode_line_segment(dash, o.linewidth, rescale))
        return data

def encode_bezier_segment(x, linewidth, rescale):
    a, c, b = x
    if is_clockwise(a, b, c):
        a, b = b, a
    len_ab = norm(b-a)
    dir_ab = (b-a) / len_ab
    rot = np.array([dir_ab, [-dir_ab[1], dir_ab[0]]])
    cc = rot.dot(c-a) / len_ab
    return [(*rescale(a), *rescale(b)), 
            (*cc, linewidth / len_ab, 2)]

def encode_spline(spline, dashing_style, linewidth, rescale):
    if not is_dashed(dashing_style):            
        return [d for curve in spline for d in encode_bezier_segment(curve, linewidth, rescale)]
    else:
        data = []
        dash_array, phase = dashing_style
        for curve in spline:
            split_curve = split_until_tolerance(curve, (0,1))
            split_length_acc = list(accumulate((bezier_length_est(x) for x, _ in split_curve), initial=0))
            total_length = split_length_acc[-1]
            split_t = [t for _, (t, _) in split_curve] + [1]
            segment_t = np.interp(list(dashed_segments(dash_array, phase, total_length)), split_length_acc, split_t)
            for t0, t1 in segment_t:
                dash = bezier_segment(curve, t0, t1)
                data.extend(encode_bezier_segment(dash, linewidth, rescale))
            phase += total_length
        return data


def encode_curve(o, rescale):
    if isinstance(o, LTLine):
        return encode_line(o, rescale)

    verts = [np.array(o.original_path[0][1])]
    segments = []
    beziers = {}
    for t, *xs in o.original_path[1:]:
        index = len(verts)-1
        if t == 'h':
            if norm(verts[0] - verts[-1]) > 1e-4:
                segments.append((index, 0))
            break
        if norm(np.array(xs[-1]) - verts[-1]) < 1e-1:
            continue
        if t == 'l':
            verts.append(np.array(xs[-1]))
            segments.append((index, index+1))
        else:
            # Each cubic bezier is approximated with two quadratic beziers
            x0 = verts[-1]
            x3 = np.array(xs[-1])
            cubic_pts = [x0] + xs if t == 'c' else \
                        [x0,x0] + xs if t == 'v' else \
                        [x0] + xs + [x3] if t == 'y' else None
            c0, mid, c1 = cubic2quads(*np.array(cubic_pts))
            for y0, y1, y2 in ((x0, c0, mid), (mid, c1, x3)):
                index = len(verts)-1
                if y1 is not None and min(norm(y1-y0), norm(y1-y2), norm(y2-y0)) > 1e-1:
                    verts.extend([y1, y2])
                    segments.append((index, index+2))
                    beziers[(index, index+2)] = index+1
                else:
                    verts.append(y2)
                    segments.append((index, index+1))

    index = len(verts)-1
    if norm(verts[0] - verts[-1]) < 1e-4:
        if segments:
            segments[-1] = (segments[-1][0], 0)
        del verts[-1]
        if (index-2, index) in beziers:
            beziers[(index-2, 0)] = beziers[(index-2, index)]
            del beziers[(index-2, index)]
    elif o.fill and t != 'h':
        segments.append((index, 0))

    verts = np.array(verts)
    data = []

    if o.stroke:
        def control_pts(s):
            a, b = verts[s[0]], verts[s[1]]
            c = verts[beziers[s]] if s in beziers else (a + b) / 2
            return a,c,b

        data.extend(encode_spline(map(control_pts, segments), o.dashing_style, o.linewidth, rescale))

    # For fill we follow this paper, but with quadratic beziers only
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/01/p1000-loop.pdf
    if o.fill and is_black(o.non_stroking_color):
        def is_intersecting(s, t):
            return not (set(s) & set(t)) and segment_intersect(*verts[list(s+t)]) 

        if any(is_intersecting(s,t) for s,t in combinations(set(segments) - set(beziers), 2)):
            print("Self-intersection detected. Skipping")
            print(o.original_path)
            return data

        def tri_segments(s):
            return combinations(s + (beziers[s],), 2) if s in beziers else [s]

        removed_vs = set()
        for _ in range(5):
            segments.sort(key=lambda s: -norm(verts[s[0]]-verts[s[1]]) if s in beziers else 1)
            replaced = set()
            new_segments = []
            new_verts = []
            index = len(verts)-1
            for j, s1 in enumerate(segments[:len(beziers)]):
                if any(is_intersecting(t1,t2) for s2 in segments[j+1:] for t1, t2 in product(tri_segments(s1), tri_segments(s2))):
                    replaced.add(s1)
                    removed_vs.add(beziers[s1])
                    c0, mid, c1 = split_bezier(*verts[[s1[0], beziers[s1], s1[1]]])
                    new_verts.extend([c0, mid, c1])
                    new_segments.extend([(s1[0], index+2), (index+2, s1[1])])
                    beziers[(s1[0], index+2)] = index+1
                    beziers[(index+2, s1[1])] = index+3
                    index += 3
            if not replaced:
                break
            verts = np.vstack([verts, new_verts])
            segments = [s for s in segments + new_segments if s not in replaced]
            beziers = {k:v for k,v in beziers.items() if k not in replaced}
        else:
            print("Possible self-intersection. Skipping")
            print(o.original_path)
            return data
        
        reordered_vs = []
        next_v = dict(segments)
        i = segments[0][0]
        while True:
            reordered_vs.append(i)
            i = next_v[i]
            if i == segments[0][0]:
                break
        verts_reordered = verts[reordered_vs]
        n = len(verts_reordered)
        max_id = max(range(n), key=lambda x: tuple(verts_reordered[x]))
        path_is_clockwise = is_clockwise(*verts_reordered[np.arange(max_id-1, max_id+2) % n])
        bv = set(reordered_vs)
        iv = set()
        ev = set()
        for (a,b),c in beziers.items():
            is_convex_fill = is_clockwise(*verts[[a,c,b]]) == path_is_clockwise
            (ev if is_convex_fill else iv).add(c)
        assert(bv | iv | ev | removed_vs == set(range(len(verts))))

        # -------------------- no winding direction information below this line --------------------

        remained_vs = list(set(range(len(verts))) - removed_vs)
        oldv2newv = {x:i for i,x in enumerate(remained_vs)}
        def olds2news(s):
            return tuple(sorted(oldv2newv[x] for x in s))

        verts = verts[remained_vs]
        bv = set(oldv2newv[x] for x in bv)
        iv = set(oldv2newv[x] for x in iv)
        ev = set(oldv2newv[x] for x in ev)
        beziers = {olds2news(s) : oldv2newv[x] for (s,x) in beziers.items()}
        segments = set(map(olds2news, segments))
        segments_all = [ss for s in segments for ss in tri_segments(s)]

        box, ext_pt = bounding_box(verts)
        verts = np.vstack([verts, box])
        ev |= set(range(len(verts)-4, len(verts)))
        t = triangle.triangulate({'vertices': verts,
                                  'segments': segments_all,
                                  'regions': [(*ext_pt, 1, 0)]}, 
                                  'pAc')
        assert(verts.shape == t['vertices'].shape)

        tris = np.sort(t['triangles'])
        chords = {s:d for x,(d,) in zip(tris, t['triangle_attributes']) for s in combinations(x, 2)
                  if set(s) <= bv and s not in segments}
        new_verts = []
        index = len(verts)-1
        split_chords = []
        for (x,y),d in chords.items():
            new_verts.append((verts[x] + verts[y]) / 2)
            index += 1
            (ev if d else iv).add(index)
            split_chords.extend([(x,index), (y,index)])
        for x in tris:
            if bv >= set(x) and segments >= set(combinations(x, 2)):
                new_verts.append(np.sum(verts[x], axis=0) / 3)
                index += 1
                iv.add(index)
        if new_verts:
            segments_all = list(set(s for x in tris for s in combinations(x, 2) if s not in chords)) + split_chords
            verts = np.vstack([verts, new_verts])
            t = triangle.triangulate({'vertices': verts,
                                      'segments': segments_all}, 
                                      'p')
            assert(verts.shape == t['vertices'].shape)
                
        for x in t['triangles']:
            z = 1
            xi, xb, xe = ((set(x) & vv) for vv in (iv, bv, ev))
            if len(xi) == 3:
                a,b,c = xi
                w = tri_uv_packed(0,0,0)
            elif len(xi) == 2 and len(xb) == 1:
                a,b = xi
                c, = xb
                w = tri_uv_packed(0,0,1)
            elif len(xi) == 1 and len(xb) == 2:
                a,b = sorted(xb)
                c, = xi
                if (a,b) not in beziers:
                    w = tri_uv_packed(1,1,0)
                elif beziers[(a,b)] == c:
                    w = -1
                    z = -1
                else:
                    w = tri_uv_packed(1,-1,0)
            elif len(xb) == 2 and len(xe) == 1:
                a,b = sorted(xb)
                c, = xe
                if (a,b) not in beziers:
                    w = tri_uv_packed(1,1,2)
                elif beziers[(a,b)] == c:
                    w = -1
                else:
                    continue
            elif len(xb) == 1 and len(xe) == 2:
                continue
            elif len(xe) == 3:
                continue
            else:
                print(f"unexpected #i, #b, #e = {[len(xi), len(xb), len(xe)]}", )
                continue
            data.extend([(*rescale(verts[a]), *rescale(verts[b])),
                         (*rescale(verts[c]), z, w)])

    return data
