import numpy as np
import matplotlib as mpl
from nuscenes.map_expansion import arcline_path_utils as APU
import hashlib

def visible(xy, r): 
    return np.any((np.abs(xy[:,0])<=r) & (np.abs(xy[:,1])<=r))

def color_for_key(key):
    h = int(hashlib.sha1(str(key).encode()).hexdigest(), 16)
    idx = h % 20
    cmap = mpl.colormaps['tab20']   # 新 API，不再有第二個參數
    return cmap((idx + 0.5) / 20.0)

def _clip_segment_length(p1, p2, L=4.0):
    """把線段以中點為中心裁成長度 L（純檢查/視覺用途）。"""
    v = p2 - p1
    d = np.linalg.norm(v)
    if d < 1e-3:
        return np.vstack([p1, p2]).astype(np.float32)
    u = v / d
    m = 0.5 * (p1 + p2)
    half = 0.5 * L
    return np.vstack([m - half*u, m + half*u]).astype(np.float32)

def report_stop_line_quality(stop_line_polylines,
                             lane_center_polylines,
                             roi_radius,
                             near_lane_r=35.0,
                             clip_L=4.0,
                             length_clip_thresh=6.0,
                             top_k=3,
                             prefix="[stop_line quality]"):
    """評估: stop_line 是否近乎垂直於附近車道切向（主程式呼叫這個）。"""
    def _nearest_tangent(poly, point):
        best_th, best_dist = 0.0, 1e9
        for i in range(len(poly)-1):
            a, b = poly[i], poly[i+1]
            ab = b - a
            t = np.clip(np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            proj = a + t * ab
            d = np.linalg.norm(point - proj)
            if d < best_dist:
                best_dist = d
                best_th  = np.arctan2(ab[1], ab[0])
        return best_th, best_dist

    def _ang_diff_abs(a, b):
        d = (a - b + np.pi) % (2*np.pi) - np.pi
        return abs(d)

    ROI = float(roi_radius) + 2.0  # 只檢視窗內（加 2m buffer）
    devs, worst = [], []

    for sl in stop_line_polylines:
        if not isinstance(sl, np.ndarray) or sl.shape[0] < 2:
            continue

        # 視窗過濾：以幾何中心判斷是否在 ROI 內
        mid_sl_full = sl.mean(axis=0)
        if (abs(mid_sl_full[0]) > ROI) or (abs(mid_sl_full[1]) > ROI):
            continue

        # 用局部（必要時裁短）的停車線方向
        seg = np.vstack([sl[0], sl[-1]]) if sl.shape[0] > 2 else sl
        if np.linalg.norm(seg[1] - seg[0]) > float(length_clip_thresh):
            seg = _clip_segment_length(seg[0], seg[1], L=float(clip_L))
        mid_sl = seg.mean(axis=0)
        th_s, _ = _nearest_tangent(seg, mid_sl)

        # 與附近車道的切向比較
        th_lane, best_score, best_dist = None, 1e9, 1e9
        for poly in lane_center_polylines:
            dmin = np.min(np.linalg.norm(poly - mid_sl, axis=1))
            if dmin > float(near_lane_r):
                continue
            th, _ = _nearest_tangent(poly, mid_sl)

            # 與垂直的偏差（弧度）
            dev_perp = min(_ang_diff_abs(th_s, th + np.pi/2),
                        _ang_diff_abs(th_s, th - np.pi/2))

            # 小權重加入距離（把距離正規化到 [0,1] 再乘上 0.15 rad ≈ 8.6°）
            score = dev_perp + 0.15 * (dmin / float(near_lane_r))

            if score < best_score:
                best_score = score
                best_dist  = dmin
                th_lane    = th

        if th_lane is None:
            continue

        dev = min(_ang_diff_abs(th_s, th_lane + np.pi/2),
                  _ang_diff_abs(th_s, th_lane - np.pi/2))
        ddeg = float(np.degrees(dev))
        devs.append(ddeg)
        worst.append((ddeg, best_dist, float(np.linalg.norm(seg[1]-seg[0])), mid_sl.copy()))

    if devs:
        devs = np.array(devs)
        p95 = np.percentile(devs, 95)
        print(f"{prefix} count={len(devs)}  mean={devs.mean():.1f}°  p95={p95:.1f}°  max={devs.max():.1f}°")
        worst.sort(reverse=True, key=lambda x: x[0])
        for k, (ang, dist, Lseg, mid) in enumerate(worst[:int(top_k)], 1):
            print(f"  worst#{k}: dev={ang:.1f}°, dist_to_lane={dist:.2f} m, "
                  f"seg_len={Lseg:.2f} m, seg_mid=({mid[0]:.1f},{mid[1]:.1f})")
    else:
        print(f"{prefix} no stop_line samples")    

# ---------- Map geometry helpers ----------
def _centerline_from_polygon_PCA(nmap, lane_rec, ds=2.0):
    # 只看 exterior ring
    poly = _polygon_xy_from_token(nmap, lane_rec.get('polygon_token', None))
    if poly is None or poly.shape[0] < 3:
        return None
    # 去掉重複閉合點
    P = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    if P.shape[0] < 3:
        return None

    # PCA：取最大方差方向
    mu = P.mean(axis=0, keepdims=True)
    X = P - mu
    # 2x2 covariance
    C = (X.T @ X) / max(len(P)-1, 1)
    try:
        vals, vecs = np.linalg.eigh(C)
    except np.linalg.LinAlgError:
        return None
    v = vecs[:, np.argmax(vals)]  # 主成分方向（單位化）
    v = v / (np.linalg.norm(v) + 1e-9)

    # 投影到主軸，取 min/max 當成線段端點，再取“中線”（其實就是通過均值的長線段）
    proj = (X @ v)   # [N]
    a, b = proj.min(), proj.max()
    # 直線中心（通過均值 mu），長度就是 b-a
    nstep = max(int(np.ceil((b - a) / ds)), 2)
    ts = np.linspace(a, b, nstep)
    line = mu + np.outer(ts, v)  # [nstep, 2]
    return line.astype(np.float32)

def _line_from_divider_token(nmap, divider_tok: str, table: str):
    """table: 'lane_divider' 或 'road_divider'。回傳對應 line polyline。"""
    if not divider_tok:
        return None
    try:
        drec = nmap.get(table, divider_tok)
        return _line_xy_from_token(nmap, drec.get('line_token'))
    except Exception:
        return None

def _stop_line_polyline(nmap, stop_token: str):
    """Parse a stop_line into an Nx2 polyline.
    優先使用 line_token/line_tokens；再退回直接點列；
    最後若是 polygon_token，取外框『最長邊』而不是對角線。
    """
    try:
        rec = nmap.get('stop_line', stop_token)
    except Exception:
        return None

    # A) line_token / line_tokens
    segs = []
    if rec.get('line_token'):
        xy = _line_xy_from_token(nmap, rec['line_token'])
        if xy is not None and xy.shape[0] >= 2:
            segs.append(xy)
    if rec.get('line_tokens'):
        for t in rec['line_tokens']:
            xy = _line_xy_from_token(nmap, t)
            if xy is not None and xy.shape[0] >= 2:
                segs.append(xy)
    if segs:
        return np.vstack(segs)

    # B) 直接點列在 stop_line 記錄上
    for k in ('points', 'coords', 'xy'):
        if rec.get(k) is not None:
            arr = np.asarray(rec[k], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
                return arr[:, :2]


    # C) polygon_token → 針對每條邊的中點作「局部」判斷：找附近多條 lane/connector 的切向，
    #     選擇與切向最垂直的那條邊；沒有車道時退回最長邊。
    if rec.get('polygon_token'):
        poly = _polygon_xy_from_token(nmap, rec['polygon_token'])
        if poly is not None and poly.shape[0] >= 3:
            P = poly
            if not np.allclose(P[0], P[-1]):
                P = np.vstack([P, P[0]])  # 關閉多邊形

            diffs = np.diff(P, axis=0)  # 每條邊向量

            def nearest_tangent(polyline, point):
                best_th, best_dist = 0.0, 1e9
                for i in range(len(polyline) - 1):
                    a, b = polyline[i], polyline[i + 1]
                    ab = b - a
                    t = np.clip(np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
                    proj = a + t * ab
                    d = np.linalg.norm(point - proj)
                    if d < best_dist:
                        best_dist = d
                        best_th = np.arctan2(ab[1], ab[0])
                return best_th, best_dist

            def ang(v): return np.arctan2(v[1], v[0])

            # 對每條「邊」單獨評分（用該邊中點 mid_e 的局部切向）
            devs_per_edge = []
            for i in range(len(diffs)):
                mid_e = 0.5 * (P[i] + P[i+1])
                th_candidates = []
                try:
                    nearby = []
                    # 半徑 30 m 內抓多條 lane / connector（最多 20 條）
                    for layer in ("lane", "lane_connector"):
                        got = nmap.get_records_in_radius(float(mid_e[0]), float(mid_e[1]), 30.0, [layer]).get(layer, [])
                        nearby.extend(got)
                    for tok in nearby[:20]:
                        xy = _lane_or_connector_centerline(nmap, tok, ds=0.5)  # 取樣稍微細一點
                        if xy is None or xy.shape[0] < 2:
                            continue
                        th, _ = nearest_tangent(xy, mid_e)
                        th_candidates.append(th)
                except Exception:
                    pass

                th_e = ang(diffs[i])
                if th_candidates:
                    # 與所有候選切向相比，取「最接近垂直」的偏差
                    dev_e = min(
                        min(abs((th_e - th + np.pi/2 + np.pi) % (2*np.pi) - np.pi),
                            abs((th_e - th - np.pi/2 + np.pi) % (2*np.pi) - np.pi))
                        for th in th_candidates
                    )
                else:
                    # 找不到車道：用一個較大的偏差，讓它落回到後面的最長邊 fallback
                    dev_e = 1e3
                devs_per_edge.append(dev_e)

            # 先看是否有合理候選（dev_e 有被填到）
            if len(devs_per_edge) and min(devs_per_edge) < 1e2:
                i = int(np.argmin(devs_per_edge))
                return P[i:i+2]

            # 沒抓到任何車道切向 → 退回最長邊（避免對角線）
            lens = np.linalg.norm(diffs, axis=1)
            i = int(np.argmax(lens))
            return P[i:i+2]
    return None

def _line_xy_from_token(nmap, line_token):
    """Return Nx2 polyline in WORLD from one 'line' token."""
    if not line_token:
        return None
    ln = nmap.get('line', line_token)

    # node-based（最常見）
    for k_nodes in ('node_tokens', 'nodes'):
        if k_nodes in ln and ln[k_nodes]:
            pts = []
            for ntok in ln[k_nodes]:
                nd = nmap.get('node', ntok)
                x = nd.get('x', nd.get('px'))
                y = nd.get('y', nd.get('py'))
                if x is None or y is None:
                    continue
                pts.append([float(x), float(y)])
            return np.asarray(pts, dtype=np.float32) if pts else None

    # 直接點列（少見）
    for k_pts in ('points', 'coords', 'xy'):
        if k_pts in ln and ln[k_pts]:
            arr = np.asarray(ln[k_pts], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]

    return None

def _lane_center_polyline(nmap, lane_token, ds=1.0):
    # A) arcline path（最理想）
    try:
        ap = nmap.get_arcline_path(lane_token)
        xy = ap.discretize(ds=ds)
        return np.asarray(xy, dtype=np.float32)
    except Exception:
        pass

    # 取得 lane 或 lane_connector 的記錄
    rec = None
    for tb in ('lane', 'lane_connector'):
        try:
            rec = nmap.get(tb, lane_token)
            break
        except KeyError:
            pass
    if rec is None:
        return None

    # B) 直接有 centerline 的情況
    if rec.get('centerline_line_token'):
        return _line_xy_from_token(nmap, rec['centerline_line_token'])
    if rec.get('line_token'):  # 有些版本中心線就叫 line_token
        return _line_xy_from_token(nmap, rec['line_token'])
    if rec.get('centerline'):
        arr = np.asarray(rec['centerline'], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]

    # C) fallback：由左右邊界平均（需要先把 divider token 轉成 line）
    Ltok = rec.get('left_lane_divider_token')  or rec.get('left_boundary_line_token')
    Rtok = rec.get('right_lane_divider_token') or rec.get('right_boundary_line_token')
    Lxy = _line_from_divider_token(nmap, Ltok, 'lane_divider') if Ltok else None
    Rxy = _line_from_divider_token(nmap, Rtok, 'lane_divider') if Rtok else None
    if Lxy is not None and Rxy is not None:
        n = min(len(Lxy), len(Rxy))
        if n >= 2:
            return 0.5 * (Lxy[:n] + Rxy[:n])
    return None

def _polygon_xy_from_token(nmap, polygon_token):
    """From a 'polygon' token, return Nx2 exterior ring in WORLD."""
    if not polygon_token:
        return None
    poly = nmap.get('polygon', polygon_token)
    nodes = poly.get('exterior_node_tokens', poly.get('exterior', None))
    if nodes is None:
        return None
    pts = []
    for ntok in nodes:
        nd = nmap.get('node', ntok)
        x = nd.get('x', nd.get('px', None))
        y = nd.get('y', nd.get('py', None))
        if x is None or y is None: continue
        pts.append([float(x), float(y)])
    return np.asarray(pts, dtype=np.float32) if pts else None

def _traffic_light_point(nmap, tl_token):
    rec = nmap.get('traffic_light', tl_token)

    # 直接座標
    for k in ('position', 'point', 'pos', 'coords'):
        if k in rec and rec[k] is not None:
            v = np.asarray(rec[k], dtype=np.float32).reshape(-1)
            if v.size >= 2:
                return np.array([[float(v[0]), float(v[1])]], dtype=np.float32)

    # 以 node 表示
    if 'node_token' in rec and rec['node_token']:
        nd = nmap.get('node', rec['node_token'])
        x = nd.get('x', nd.get('px')); y = nd.get('y', nd.get('py'))
        if x is not None and y is not None:
            return np.array([[float(x), float(y)]], dtype=np.float32)

    # 退一步：若是 line / polygon，取中點 / 質心
    for k in ('line_token','line_tokens'):
        if k in rec and rec[k]:
            toks = [rec[k]] if isinstance(rec[k], str) else list(rec[k])
            segs = [ _line_xy_from_token(nmap, t) for t in toks ]
            segs = [ s for s in segs if s is not None and s.shape[0] >= 2 ]
            if segs:
                xy = np.vstack(segs)
                mid = xy[xy.shape[0]//2]
                return mid[None, :].astype(np.float32)

    if 'polygon_token' in rec and rec['polygon_token']:
        poly = nmap.get('polygon', rec['polygon_token'])
        nodes = poly.get('exterior_node_tokens', poly.get('exterior'))
        if nodes:
            pts = []
            for nt in nodes:
                nd = nmap.get('node', nt)
                x = nd.get('x', nd.get('px')); y = nd.get('y', nd.get('py'))
                if x is not None and y is not None:
                    pts.append([float(x), float(y)])
            if pts:
                P = np.asarray(pts, dtype=np.float32)
                # 多邊形質心（簡單版）
                return np.mean(P, axis=0, keepdims=True)
    return None

def _resample_polyline(xy: np.ndarray, ds: float = 1.0) -> np.ndarray:
    """Resample polyline to roughly uniform spacing ds (meters)."""
    if xy is None or xy.shape[0] < 2:
        return None
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    L = d[-1]
    if L < 1e-3:
        return xy[:2]
    s = np.arange(0.0, L + ds, ds)
    s[-1] = L
    out = []
    j = 0
    for si in s:
        while j < len(d) - 1 and d[j+1] < si:
            j += 1
        t = 0.0 if d[j+1] == d[j] else (si - d[j]) / (d[j+1] - d[j])
        out.append(xy[j] * (1 - t) + xy[j+1] * t)
    return np.asarray(out, dtype=np.float32)

def _xy_from_any_points(v) -> np.ndarray:
    """
    將多種格式的點列轉成 [N,2] float32：
    - [[x,y], ...] 或 [[x,y,z], ...]
    - [{'x':..,'y':..}, ...]
    """
    if v is None:
        return None
    arr = np.asarray(v)
    # 物件陣列（list of dict）
    if isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], dict):
        pts = []
        for d in v:
            if 'x' in d and 'y' in d:
                pts.append([float(d['x']), float(d['y'])])
        return np.asarray(pts, dtype=np.float32) if pts else None
    # 一般 array
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].astype(np.float32)
    return None

def _lane_center_from_boundaries(nmap, lane_token: str, ds: float = 1.0) -> np.ndarray:
    """
    lane 讀不到中心線時，用左右 lane_divider 的折線平均出 centerline。
    支援 v1.3 常見欄位：left/right_lane_divider(_token), left/right_boundary_line_token
    """
    lane = nmap.get('lane', lane_token)
    Ltok = (lane.get('left_lane_divider_token')  or lane.get('left_lane_divider')
            or lane.get('left_boundary_line_token'))
    Rtok = (lane.get('right_lane_divider_token') or lane.get('right_lane_divider')
            or lane.get('right_boundary_line_token'))

    def poly_from_div(tok):
        if not tok: return None
        rec = nmap.get('lane_divider', tok)
        return _line_xy_from_token(nmap, rec.get('line_token'))

    L = poly_from_div(Ltok); R = poly_from_div(Rtok)
    if L is None or R is None or L.shape[0] < 2 or R.shape[0] < 2:
        return None

    Lr = _resample_polyline(L, ds); Rr = _resample_polyline(R, ds)
    n = min(Lr.shape[0], Rr.shape[0])
    if n < 2: return None
    return (0.5 * (Lr[:n] + Rr[:n])).astype(np.float32)

def _nodes_to_xy(nmap, node_tokens):
    pts = []
    for nt in node_tokens:
        nd = nmap.get('node', nt)
        x = nd.get('x', nd.get('px')); y = nd.get('y', nd.get('py'))
        if x is None or y is None: continue
        pts.append([float(x), float(y)])
    return np.asarray(pts, dtype=np.float32) if pts else None

def _poly_from_segment_nodes(nmap, seg_nodes_field):
    """
    seg_nodes_field: list[list[node_token]], 例如 lane['left_lane_divider_segment_nodes']
    """
    if not isinstance(seg_nodes_field, list) or len(seg_nodes_field) == 0:
        return None
    parts = []
    for seg in seg_nodes_field:
        xy = _nodes_to_xy(nmap, seg)
        if xy is not None and xy.shape[0] >= 2:
            parts.append(xy)
    if not parts:
        return None
    return _concat_polys(parts)

def _poly_from_segments(nmap, seg_field):
    """
    seg_field: list[line_token] 或 list[lane_divider_token]（依 JSON 而定）
    會自動判斷 token 屬於哪個 table，最後取得對應 line 的 polyline。
    """
    if not isinstance(seg_field, list) or len(seg_field) == 0:
        return None
    parts = []
    for tok in seg_field:
        xy = None
        # 先試 line
        try:
            nmap.get('line', tok)  # 只是測試 token 是否屬於這張表
            xy = _line_xy_from_token(nmap, tok)
        except Exception:
            # 再試 lane_divider -> line_token
            try:
                drec = nmap.get('lane_divider', tok)
                xy = _line_xy_from_token(nmap, drec.get('line_token', None))
            except Exception:
                xy = None
        if xy is not None and xy.shape[0] >= 2:
            parts.append(xy)
    if not parts:
        return None
    return _concat_polys(parts)

def _lane_center_from_boundaries_v13(nmap, lane_rec, ds=1.0):
    """
    v1.3 這種 schema：用左右邊界（segment_nodes 或 segments）平均出 centerline。
    """
    # 1) 先用 segment_nodes
    L = _poly_from_segment_nodes(nmap, lane_rec.get('left_lane_divider_segment_nodes'))
    R = _poly_from_segment_nodes(nmap, lane_rec.get('right_lane_divider_segment_nodes'))

    # 2) 若沒有，再用 segments（可能是 line_token 或 lane_divider_token 的 list）
    if L is None:
        L = _poly_from_segments(nmap, lane_rec.get('left_lane_divider_segments'))
    if R is None:
        R = _poly_from_segments(nmap, lane_rec.get('right_lane_divider_segments'))

    if L is None or R is None or L.shape[0] < 2 or R.shape[0] < 2:
        return None

    Lr = _resample_polyline(L, ds); Rr = _resample_polyline(R, ds)
    n = min(Lr.shape[0], Rr.shape[0])
    if n < 2:
        return None
    return (0.5 * (Lr[:n] + Rr[:n])).astype(np.float32)

def _polyline_from_any_line_field(nmap, rec) -> np.ndarray:
    """
    兼容舊版寫法；這裡只保留最常見幾種欄位，其他交給 v1.3 的分支。
    """
    for k in ('baseline_path', 'centerline', 'points', 'coords', 'xy'):
        if k in rec and rec[k] is not None:
            a = np.asarray(rec[k], dtype=np.float32)
            if a.ndim == 2 and a.shape[1] >= 2 and a.shape[0] >= 2:
                return a[:, :2]
    for k in ('centerline_line_token', 'line_token'):
        t = rec.get(k)
        if isinstance(t, str) and t:
            xy = _line_xy_from_token(nmap, t)
            if xy is not None and xy.shape[0] >= 2:
                return xy
    if isinstance(rec.get('line_tokens', None), list):
        pts = []
        for t in rec['line_tokens']:
            xy = _line_xy_from_token(nmap, t)
            if xy is not None and xy.shape[0] >= 2:
                pts.append(xy)
        if pts:
            return _concat_polys(pts)
    return None

def _fetch_node_xy(nmap, node_ref):
    """node_ref 可以是 token(str) / 索引(int) / dict({'x','y'} 或 {'node_token':...})。"""
    try:
        if isinstance(node_ref, str):
            nd = nmap.get('node', node_ref)
        elif isinstance(node_ref, int):
            nd = nmap.node[node_ref]            # 直接用 index 取
        elif isinstance(node_ref, dict):
            if 'x' in node_ref and 'y' in node_ref:
                return float(node_ref['x']), float(node_ref['y'])
            if 'px' in node_ref and 'py' in node_ref:
                return float(node_ref['px']), float(node_ref['py'])
            if 'node_token' in node_ref:
                nd = nmap.get('node', node_ref['node_token'])
            else:
                return None
        else:
            return None
        x = nd.get('x', nd.get('px', None)); y = nd.get('y', nd.get('py', None))
        if x is None or y is None: return None
        return float(x), float(y)
    except Exception:
        return None

def _nodeslist_to_xy(nmap, nodes_list):
    """nodes_list 是 node 參考的 list（token / index / dict 混合亦可）。"""
    pts = []
    for ref in nodes_list:
        p = _fetch_node_xy(nmap, ref)
        if p is None: continue
        pts.append(p)
    return np.asarray(pts, dtype=np.float32) if len(pts) >= 2 else None

def _concat_polys(polys):
    out = []
    for i, p in enumerate(polys):
        if p is None or p.shape[0] == 0: continue
        if out and np.allclose(out[-1], p[0]):  # 去掉重複接點
            out.extend(p[1:])
        else:
            out.extend(p)
    return np.asarray(out, dtype=np.float32) if out else None

def _poly_from_any_segment(nmap, seg):
    """
    把「一個 segment 表示法」轉成 Nx2：
    - list/tuple -> 視為 node 序列（可混 token / index / dict）
    - dict       -> 可能含 node_tokens / nodes / line_token / points
    - str token  -> 依序嘗試 line / lane_divider / lane_divider_segment /
                    road_divider / road_divider_segment，再取 line 或 nodes
    """
    # list: nodes
    if isinstance(seg, (list, tuple)):
        return _nodeslist_to_xy(nmap, seg)

    # dict: 可能有 node_tokens / nodes / line_token / points
    if isinstance(seg, dict):
        if 'node_tokens' in seg or 'nodes' in seg:
            return _nodeslist_to_xy(nmap, seg.get('node_tokens', seg.get('nodes')))
        if 'line_token' in seg:
            return _line_xy_from_token(nmap, seg['line_token'])
        for k in ('points', 'coords', 'xy'):
            if k in seg and seg[k] is not None:
                arr = np.asarray(seg[k], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
                    return arr[:, :2]
        return None

    # str: token -> 依序嘗試幾個表
    if isinstance(seg, str):
        # 1) 直接當 line
        try:
            nmap.get('line', seg)
            return _line_xy_from_token(nmap, seg)
        except Exception:
            pass
        # 2) lane_divider / road_divider
        for table in ('lane_divider', 'road_divider'):
            try:
                drec = nmap.get(table, seg)
                return _line_xy_from_token(nmap, drec.get('line_token', None))
            except Exception:
                pass
        # 3) lane_divider_segment / road_divider_segment  (v1.3 常見)
        for table in ('lane_divider_segment', 'road_divider_segment'):
            try:
                srec = nmap.get(table, seg)
                # 優先 line_token；沒有就看 node_tokens/nodes
                if 'line_token' in srec and srec['line_token']:
                    return _line_xy_from_token(nmap, srec['line_token'])
                if 'node_tokens' in srec or 'nodes' in srec:
                    return _nodeslist_to_xy(nmap, srec.get('node_tokens', srec.get('nodes')))
            except Exception:
                pass
        return None

    return None

def _polys_from_segments_field(nmap, field):
    """field 是 list[segment]，每個 segment 可能是上面任一型態。回傳拼接後的 polyline。"""
    if not isinstance(field, list) or not field:
        return None
    parts = []
    for seg in field:
        xy = _poly_from_any_segment(nmap, seg)
        if xy is not None and xy.shape[0] >= 2:
            parts.append(xy)
    return _concat_polys(parts) if parts else None

def _lane_center_from_v13_boundaries(nmap, lane_rec, ds=1.0):
    # 先用 *_segment_nodes
    L = _polys_from_segments_field(nmap, lane_rec.get('left_lane_divider_segment_nodes'))
    R = _polys_from_segments_field(nmap, lane_rec.get('right_lane_divider_segment_nodes'))

    # 如果是空的，再用 *_segments（v1.3 很常只有這個有東西）
    if L is None or R is None:
        if L is None:
            L = _polys_from_segments_field(nmap, lane_rec.get('left_lane_divider_segments'))
        if R is None:
            R = _polys_from_segments_field(nmap, lane_rec.get('right_lane_divider_segments'))

    if L is None or R is None or L.shape[0] < 2 or R.shape[0] < 2:
        return None

    Lr = _resample_polyline(L, ds); Rr = _resample_polyline(R, ds)
    n = min(Lr.shape[0], Rr.shape[0])
    if n < 2: return None
    return (0.5 * (Lr[:n] + Rr[:n])).astype(np.float32)

def _lane_or_connector_centerline(nmap, token: str, ds: float = 1.0) -> np.ndarray:
    # 1) arcline（若有）
    try:
        ap = nmap.get_arcline_path(token)
        xy = np.asarray(APU.discretize_lane(ap, resolution_meters=ds), dtype=np.float32)
        if xy.ndim == 1:
            xy = xy[None, :]
        if xy.shape[1] >= 2:
            xy = xy[:, :2]  # 關鍵：只取 XY
        if xy.shape[0] >= 2:
            return _resample_polyline(xy, ds=ds)  # 等距
    except Exception:
        print(f"No _lane_or_connector_centerline(): _resample_polyline()")
        pass

    # 2) lane：通用欄位；失敗用 v1.3 邊界
    try:
        rec = nmap.get('lane', token)
        xy = _polyline_from_any_line_field(nmap, rec)
        if xy is None:
            xy = _lane_center_from_v13_boundaries(nmap, rec, ds=ds)
        if xy is not None and xy.shape[0] >= 2:
            return _resample_polyline(xy, ds=ds)
    except KeyError:
        pass

    # 3) lane_connector：通用欄位；再嘗試 v1.3 風格（有些 connector 也用 segment）
    try:
        rec = nmap.get('lane_connector', token)
        xy = _polyline_from_any_line_field(nmap, rec)
        if xy is None:
            xy = _lane_center_from_v13_boundaries(nmap, rec, ds=ds)
        if xy is not None and xy.shape[0] >= 2:
            return _resample_polyline(xy, ds=ds)
    except KeyError:
        pass

    # ... 先前幾個分支都不行時，最後用 lane polygon 的 PCA 當近似中心線
    try:
        rec = nmap.get('lane', token)
        xy = _centerline_from_polygon_PCA(nmap, rec, ds=2.0)
        if xy is not None and xy.shape[0] >= 2:
            return xy
    except Exception:
        pass

    return None

def extract_map_vectors(nmap, map_tokens: dict, world_to_ego_xy_fn):
    """把附近 HD map tokens 轉成 ego@t0 的向量集合（object arrays）。"""
    lanes = []
    for tok in (map_tokens.get("lane_center", []) + map_tokens.get("lane_connector", [])):
        xy = _lane_or_connector_centerline(nmap, tok, ds=1.0)
        if xy is not None and xy.shape[0] >= 2:
            lanes.append(world_to_ego_xy_fn(xy))

    lane_div, road_div, ped, stop = [], [], [], []
    for tok in map_tokens.get("lane_divider", []):
        rec = nmap.get("lane_divider", tok)
        xy = _line_xy_from_token(nmap, rec.get("line_token", None))
        if xy is not None and xy.shape[0] >= 2:
            lane_div.append(world_to_ego_xy_fn(xy))

    for tok in map_tokens.get("road_divider", []):
        rec = nmap.get("road_divider", tok)
        xy = _line_xy_from_token(nmap, rec.get("line_token", None))
        if xy is not None and xy.shape[0] >= 2:
            road_div.append(world_to_ego_xy_fn(xy))

    for tok in map_tokens.get("ped_crossing", []):
        rec = nmap.get("ped_crossing", tok)
        xy = _polygon_xy_from_token(nmap, rec.get("polygon_token", None))
        if xy is not None and xy.shape[0] >= 3:
            ped.append(world_to_ego_xy_fn(xy))

    for tok in map_tokens.get("stop_line", []):
        xy = _stop_line_polyline(nmap, tok)
        if xy is not None and xy.shape[0] >= 2:
            stop.append(world_to_ego_xy_fn(xy))

    tl_pts = []
    for tok in map_tokens.get("traffic_light", []):
        xy = _traffic_light_point(nmap, tok)
        if xy is not None:
            tl_pts.append(world_to_ego_xy_fn(xy))
    tl_pts = np.vstack(tl_pts) if len(tl_pts) else np.zeros((0, 2), np.float32)

    return {
        "lane_center":   np.array(lanes,     dtype=object),
        "lane_divider":  np.array(lane_div,  dtype=object),
        "road_divider":  np.array(road_div,  dtype=object),
        "ped_crossing":  np.array(ped,       dtype=object),
        "stop_line":     np.array(stop,      dtype=object),
        "traffic_light": tl_pts,
    }