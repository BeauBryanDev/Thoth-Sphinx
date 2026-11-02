"""Ported from the spatial_logic.py __main__ smoke tests (14 groups).

All synthetic — no ONNX model or image files needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from vision.spatial_logic import (
    CARTOUCHE_CLASS,
    Detection,
    assemble_reading_order,
    cartouche_reentry,
    cluster_quadrats,
    inset_bbox,
    letterbox,
    merge_duplicate_boxes,
    postprocess_onnx,
    tag_cartouche_members,
)

NAMES = ['Aa1', 'cartouche', 'g17', 'n35', 'd21', 'unknown']


def det(x1, y1, x2, y2, cls_, score=0.8):
    return Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                     top3=[(cls_, score), ('z1', 0.05), ('Aa1', 0.02)])


def test_detection_properties():
    d = Detection(bbox=(10.0, 20.0, 50.0, 100.0),
                  top3=[('g17', 0.9), ('n35', 0.05), ('unknown', 0.01)])
    assert d.centroid == (30.0, 60.0)
    assert d.width == 40.0 and d.height == 80.0 and d.area == 3200.0
    assert d.cls == 'g17' and d.max_score == 0.9
    assert not d.is_cartouche()


def test_postprocess_onnx_conf_filter_nms_top3():
    C, N = len(NAMES), 6
    raw = np.zeros((4 + C, N), dtype=np.float32)
    raw[4:, :] = 0.01
    # anchor 0: g17 at (100,100) 40x40, score 0.90 (+ alternates)
    raw[:4, 0] = [100, 100, 40, 40]; raw[4 + 2, 0] = 0.90
    raw[4 + 3, 0] = 0.30; raw[4 + 5, 0] = 0.10
    # anchor 1: shifted duplicate, lower score -> NMS'd
    raw[:4, 1] = [104, 102, 40, 40]; raw[4 + 2, 1] = 0.70
    # anchor 2: n35 far away
    raw[:4, 2] = [300, 100, 30, 30]; raw[4 + 3, 2] = 0.60
    # anchor 3: below conf threshold
    raw[:4, 3] = [500, 100, 30, 30]; raw[4 + 0, 3] = 0.10

    dets = postprocess_onnx(raw[None], NAMES, conf_thresh=0.25)
    assert len(dets) == 2
    assert dets[0].cls == 'g17' and abs(dets[0].max_score - 0.90) < 1e-6
    assert dets[1].cls == 'n35'
    assert [c for c, _ in dets[0].top3] == ['g17', 'n35', 'unknown']
    assert dets[0].bbox == (80.0, 80.0, 120.0, 120.0)


def test_tag_cartouche_members():
    cart = Detection(bbox=(100.0, 50.0, 300.0, 120.0),
                     top3=[('cartouche', 0.80), ('Aa1', 0.05), ('g17', 0.02)])
    inside = Detection(bbox=(120.0, 60.0, 150.0, 110.0),
                       top3=[('g17', 0.85), ('n35', 0.1), ('d21', 0.02)])
    # centroid x=302 — just past raw edge, caught by 10% expansion
    bracket = Detection(bbox=(294.0, 60.0, 310.0, 110.0),
                        top3=[('n35', 0.70), ('g17', 0.1), ('d21', 0.05)])
    outside = Detection(bbox=(400.0, 60.0, 430.0, 110.0),
                        top3=[('d21', 0.90), ('g17', 0.05), ('n35', 0.02)])
    dets = [cart, inside, bracket, outside]
    assert tag_cartouche_members(dets) == [0]
    assert inside.inside_cartouche and inside.cartouche_id == 0
    assert bracket.inside_cartouche and bracket.cartouche_id == 0
    assert not outside.inside_cartouche
    assert not cart.inside_cartouche


def test_adjacent_outer_signs_not_swallowed_by_expansion():
    """padding_cartouche_1 regression: the two X1 of nsw-bity sat just
    above the cartouche; the 10% expansion swallowed them out of the
    outer text. Membership now also requires >=20% overlap with the RAW
    cartouche bbox."""
    cart = Detection(bbox=(134.0, 256.0, 308.0, 552.0),
                     top3=[('cartouche', 0.84), ('Aa1', 0.02), ('g17', 0.01)])
    # real geometry from the failing image: bottoms ~2 px into the box
    x1_r = Detection(bbox=(247.0, 227.0, 294.0, 258.0),
                     top3=[('x1', 0.83), ('z1', 0.05), ('Aa1', 0.02)])
    x1_l = Detection(bbox=(154.0, 230.0, 198.0, 257.0),
                     top3=[('x1', 0.80), ('z1', 0.05), ('Aa1', 0.02)])
    inside = Detection(bbox=(181.0, 270.0, 258.0, 341.0),
                       top3=[('n5', 0.86), ('n33', 0.05), ('Aa1', 0.02)])
    dets = [cart, x1_r, x1_l, inside]
    tag_cartouche_members(dets, img_w=450, img_h=660)
    assert not x1_r.inside_cartouche
    assert not x1_l.inside_cartouche
    assert inside.inside_cartouche and inside.cartouche_id == 0


def test_cartouche_reentry_two_pass():
    """The standalone function still works (pipeline no longer calls it)."""
    img = np.zeros((200, 500, 3), dtype=np.uint8)
    cart = Detection(bbox=(100.0, 50.0, 300.0, 150.0),
                     top3=[('cartouche', 0.9), ('Aa1', 0.02), ('g17', 0.01)])
    lone = Detection(bbox=(130.0, 70.0, 160.0, 130.0),
                     top3=[('g17', 0.55), ('n35', 0.2), ('d21', 0.1)])
    dets = [cart, lone]
    idxs = tag_cartouche_members(dets)
    assert lone.inside_cartouche

    calls = []

    def mock_infer(crop):
        calls.append(crop.shape)
        return [
            Detection(bbox=(0.0, 0.0, 160.0, 88.0),          # bracket leak
                      top3=[('cartouche', 0.6), ('Aa1', 0.1), ('g17', 0.05)]),
            Detection(bbox=(9.0, 13.0, 41.0, 75.0),          # dup of lone
                      top3=[('g17', 0.80), ('n35', 0.1), ('d21', 0.05)]),
            Detection(bbox=(82.0, 15.0, 112.0, 75.0),        # new sign
                      top3=[('n35', 0.75), ('g17', 0.1), ('d21', 0.05)]),
        ]

    out = cartouche_reentry(img, dets, idxs, mock_infer)
    assert len(calls) == 2
    assert calls[0] == (88, 176, 3) and calls[1] == (88, 160, 3)
    assert len(out) == 3
    assert abs(lone.max_score - 0.80) < 1e-6
    assert lone.bbox == (121.0, 69.0, 153.0, 131.0)
    new = out[2]
    assert new.from_reentry and new.inside_cartouche and new.cartouche_id == 0
    assert new.bbox == (194.0, 71.0, 224.0, 131.0)
    assert all(d.cls != CARTOUCHE_CLASS or d is cart for d in out)


def test_cartouche_reentry_legacy_mode():
    img = np.zeros((200, 500, 3), dtype=np.uint8)
    cart = Detection(bbox=(100.0, 50.0, 300.0, 150.0),
                     top3=[('cartouche', 0.9), ('Aa1', 0.02), ('g17', 0.01)])
    m1 = Detection(bbox=(120.0, 70.0, 150.0, 130.0),
                   top3=[('g17', 0.8), ('n35', 0.1), ('d21', 0.05)])
    m2 = Detection(bbox=(180.0, 70.0, 210.0, 130.0),
                   top3=[('n35', 0.7), ('g17', 0.1), ('d21', 0.05)])
    dets = [cart, m1, m2]
    idxs = tag_cartouche_members(dets)

    def must_not_run(crop):
        raise AssertionError('legacy mode must not fire: 2 members tagged')

    out = cartouche_reentry(img, dets, idxs, must_not_run, always=False)
    assert len(out) == 3

    fired = []
    cartouche_reentry(img, [cart, m1, m2], idxs,
                      lambda c: fired.append(1) or [])
    assert fired, 'always=True must re-enter even populated cartouches'


def test_inset_and_letterbox_roundtrip():
    box = inset_bbox((100.0, 50.0, 160.0, 250.0), 0.06, 0.10)
    assert box == (103.6, 70.0, 156.4, 230.0)
    crop = np.zeros((300, 100, 3), dtype=np.uint8)
    canvas, scale, dx, dy = letterbox(crop, imgsz=1024)
    assert canvas.shape == (1024, 1024, 3)
    mx, my = 50 * scale + dx, 150 * scale + dy
    assert abs((mx - dx) / scale - 50) < 1e-9
    assert abs((my - dy) / scale - 150) < 1e-9


def test_merge_duplicate_boxes():
    cart = Detection(bbox=(80.0, 80.0, 180.0, 420.0),
                     top3=[('cartouche', 0.9), ('Aa1', 0.02), ('g17', 0.01)])
    s29 = Detection(bbox=(100.0, 100.0, 130.0, 200.0),
                    top3=[('s29', 0.79), ('o34', 0.05), ('f31', 0.04)],
                    inside_cartouche=True, cartouche_id=0, from_reentry=True)
    f31 = Detection(bbox=(105.0, 102.0, 122.0, 198.0),
                    top3=[('f31', 0.50), ('m17', 0.10), ('z4', 0.03)],
                    inside_cartouche=True, cartouche_id=0, from_reentry=True)
    far = Detection(bbox=(100.0, 300.0, 130.0, 400.0),
                    top3=[('n35', 0.60), ('n37', 0.05), ('z7', 0.02)],
                    inside_cartouche=True, cartouche_id=0, from_reentry=True)
    merged = merge_duplicate_boxes([cart, s29, f31, far])
    assert len(merged) == 3
    assert merged[1] is s29
    assert [c for c, _ in s29.top3] == ['s29', 'f31', 'm17']
    assert abs(dict(s29.top3)['f31'] - 0.50) < 1e-9
    assert merged[0] is cart
    assert s29.cartouche_id == 0 and far.cartouche_id == 0


def test_cluster_quadrats_columns():
    n35q = det(260, 60, 330, 85, 'n35', 0.6)
    m17q = det(300, 100, 315, 180, 'm17', 0.65)
    s29q = det(265, 100, 290, 180, 's29', 0.79)
    x1q = det(265, 200, 325, 225, 'x1', 0.7)
    qs = cluster_quadrats([n35q, m17q, s29q, x1q], layout='columns')
    assert sorted(len(q.members) for q in qs) == [1, 1, 2]
    pair = next(q for q in qs if len(q.members) == 2)
    assert {d.cls for d in pair.members} == {'m17', 's29'}
    assert [d.cls for d in pair.ordered('rtl')] == ['m17', 's29']
    assert [d.cls for d in pair.ordered('ltr')] == ['s29', 'm17']


def test_cluster_quadrats_rows():
    a = det(100, 100, 140, 140, 'g17')
    b = det(150, 100, 190, 140, 'd21')
    c = det(200, 100, 240, 140, 'x1')
    top = det(250, 95, 290, 115, 'n35', 0.7)
    bot = det(250, 122, 290, 142, 'z1', 0.6)
    qs = cluster_quadrats([a, b, c, top, bot], layout='rows')
    assert sorted(len(q.members) for q in qs) == [1, 1, 1, 2]
    pair = next(q for q in qs if len(q.members) == 2)
    assert [d.cls for d in pair.ordered('rtl')] == ['n35', 'z1']


def test_oversized_component_splits_at_largest_gap():
    run = [det(x, 100, x + 14, 180, 'm17', 0.6)
           for x in (0, 20, 40, 75, 95)]           # largest gap 40->75
    qs = cluster_quadrats(run, layout='columns', max_signs=4)
    assert sorted(len(q.members) for q in qs) == [2, 3]


def test_assemble_columns_rtl_and_boundary_hints():
    rg, rn = det(300, 100, 340, 140, 'g17'), det(300, 160, 340, 200, 'n35')
    ld, lx = det(100, 100, 140, 140, 'd21'), det(100, 160, 140, 200, 'x1')
    quads = cluster_quadrats([rg, rn, ld, lx], layout='columns')
    ro = assemble_reading_order(quads, layout='columns', direction='rtl')
    assert [s[0][0] for s in ro.slots] == ['g17', 'n35', 'd21', 'x1']
    assert ro.boundary_hints == [2, 4]
    assert ro.n_synthetic == 0
    ro2 = assemble_reading_order(quads, layout='columns', direction='ltr')
    assert [s[0][0] for s in ro2.slots] == ['d21', 'x1', 'g17', 'n35']


def test_synthetic_unknown_insertion():
    t = det(100, 100, 140, 130, 'n35')
    btm = det(100, 180, 140, 210, 's29')
    quads = cluster_quadrats([t, btm], layout='columns')
    ro = assemble_reading_order(quads, layout='columns', direction='rtl')
    assert [s[0][0] for s in ro.slots] == ['n35', 'unknown', 's29']
    assert ro.slot_detections[1] is None
    assert ro.slots[1] == [('unknown', 0.0)]
    # leading-edge gap with extent (the Unas e34 case)
    ro = assemble_reading_order(quads, layout='columns', direction='rtl',
                                extent=(95.0, 50.0, 145.0, 240.0))
    assert [s[0][0] for s in ro.slots] == ['unknown', 'n35', 'unknown', 's29']
    # huge gap capped at MAX_GAP_INSERTS
    big = det(100, 330, 140, 360, 'x1')
    quads = cluster_quadrats([t, big], layout='columns')
    ro = assemble_reading_order(quads, layout='columns', direction='rtl')
    assert [s[0][0] for s in ro.slots] == ['n35', 'unknown', 'unknown', 'x1']


def test_rows_rtl_order_and_gap():
    r1a, r1b = det(300, 100, 340, 140, 'g17'), det(250, 100, 290, 140, 'd21')
    r2a = det(300, 200, 340, 240, 'x1')
    quads = cluster_quadrats([r1a, r1b, r2a], layout='rows')
    ro = assemble_reading_order(quads, layout='rows', direction='rtl')
    assert [s[0][0] for s in ro.slots] == ['g17', 'd21', 'x1']
    assert ro.boundary_hints == [2, 3]
    quads = cluster_quadrats([r1a, det(190, 100, 230, 140, 'd21')],
                             layout='rows')
    ro = assemble_reading_order(quads, layout='rows', direction='rtl')
    assert [s[0][0] for s in ro.slots] == ['g17', 'unknown', 'd21']


def test_single_line_forces_one_column():
    n_ = det(140, 100, 160, 120, 'n35')
    m_ = det(100, 140, 112, 170, 'm17')
    s_ = det(118, 140, 143, 170, 's29')
    quads = cluster_quadrats([n_, m_, s_], layout='columns')
    ext = (95.0, 60.0, 160.0, 200.0)
    multi = assemble_reading_order(quads, layout='columns', direction='rtl',
                                   extent=ext)
    single = assemble_reading_order(quads, layout='columns', direction='rtl',
                                    extent=ext, single_line=True)
    assert len(multi.lines) == 2           # the failure mode: fake columns
    assert len(single.lines) == 1
    assert [s[0][0] for s in single.slots] == ['unknown', 'n35', 's29', 'm17']
