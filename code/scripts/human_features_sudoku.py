# human_features_sudoku.py

from __future__ import annotations
from typing import List, Tuple, Dict, Set
import numpy as np
import math


def peers_of(r: int, c: int) -> List[Tuple[int,int]]:
    P = set()
    for cc in range(9): P.add((r,cc))
    for rr in range(9): P.add((rr,c))
    br, bc = 3*(r//3), 3*(c//3)
    for rr in range(br, br+3):
        for cc in range(bc, bc+3):
            P.add((rr,cc))
    P.discard((r,c))
    return list(P)

def candidates_for_cell(grid: np.ndarray, r: int, c: int) -> Set[int]:
    if grid[r,c] != 0: 
        return set()
    row = set(grid[r, :]) - {0}
    col = set(grid[:, c]) - {0}
    br, bc = 3*(r//3), 3*(c//3)
    box = set(grid[br:br+3, bc:bc+3].reshape(-1)) - {0}
    used = row | col | box
    return set(range(1,10)) - used

def all_candidates(grid: np.ndarray) -> Dict[Tuple[int,int], Set[int]]:
    C = {}
    empties = np.argwhere(grid == 0)
    for (r,c) in map(tuple, empties):
        C[(r,c)] = candidates_for_cell(grid, r, c)
    return C

def box_index(r:int,c:int) -> int:
    return (r//3)*3 + (c//3)

# --- агрегаты по юнитам ---

def unit_cells_row(r:int) -> List[Tuple[int,int]]:
    return [(r, c) for c in range(9)]

def unit_cells_col(c:int) -> List[Tuple[int,int]]:
    return [(r, c) for r in range(9)]

def unit_cells_box(b:int) -> List[Tuple[int,int]]:
    br, bc = 3*(b//3), 3*(b%3)
    return [(r, c) for r in range(br, br+3) for c in range(bc, bc+3)]

def count_by_size(cset: List[Set[int]]) -> Tuple[int,int,int,int]:
    """сколько клеток с 1/2/3/≥4 кандидатами среди переданного списка"""
    ones=twos=threes=ge4=0
    for S in cset:
        k=len(S)
        if k==1: ones+=1
        elif k==2: twos+=1
        elif k==3: threes+=1
        elif k>=4: ge4+=1
    return ones, twos, threes, ge4

def entropy_from_counts(counts: List[int]) -> float:
    s = sum(counts)
    if s <= 0: return 0.0
    ent = 0.0
    for x in counts:
        if x>0:
            p = x/s
            ent -= p*math.log(p+1e-12)
    return ent

# --- «человеческие» паттерны по юниту ---

def naked_pair_triple_flags(cset: List[Set[int]]) -> Tuple[int,int,int]:
    """
    Возвращает индикаторы:
      - есть naked pair?  (ровно 2 клетки имеют ровно одинаковый набор из 2 кандидатов)
      - есть naked triple? (ровно 3 клетки с одинаковым набором из 3 кандидатов)
      - есть «fishy» (разные пары/тройные пересекаются) — грубая эвристика 0/1
    """
    from collections import Counter
    pairs = [tuple(sorted(S)) for S in cset if len(S)==2]
    triples= [tuple(sorted(S)) for S in cset if len(S)==3]
    cnt_p  = Counter(pairs)
    cnt_t  = Counter(triples)
    has_pair   = int(any(v==2 for v in cnt_p.values()))
    has_triple = int(any(v==3 for v in cnt_t.values()))
    fishy = int(len([1 for v in cnt_p.values() if v==2])>=2 or
                len([1 for v in cnt_t.values() if v==3])>=2)
    return has_pair, has_triple, fishy

def hidden_single_count_for_digit(cset: List[Set[int]], d: int) -> int:
    """сколько клеток юнита допускают именно d (для hidden single: ==1)"""
    places = sum(1 for S in cset if d in S)
    return places

def pointing_claiming_flags(cands: Dict[Tuple[int,int], Set[int]], d:int, r:int, c:int) -> Tuple[int,int,int,int]:
    """
    Эвристики box-line:
    - pointing row/col: в боксе кандидаты d только в одной строке/столбце;
    - claiming row/col: в строке/столбце кандидаты d только в одном боксе.
    Возвращаем 4 индикатора (0/1): pointing_row, pointing_col, claiming_row, claiming_col
    """
    b = box_index(r,c)
    box_cells = unit_cells_box(b)
    rows_in_box = set()
    cols_in_box = set()
    for (rr,cc) in box_cells:
        if d in cands.get((rr,cc), set()):
            rows_in_box.add(rr)
            cols_in_box.add(cc)
    pointing_row = int(len(rows_in_box)==1)
    pointing_col = int(len(cols_in_box)==1)

    row_cells = unit_cells_row(r)
    col_cells = unit_cells_col(c)
    boxes_in_row = set()
    for (rr,cc) in row_cells:
        if d in cands.get((rr,cc), set()):
            boxes_in_row.add(box_index(rr,cc))
    boxes_in_col = set()
    for (rr,cc) in col_cells:
        if d in cands.get((rr,cc), set()):
            boxes_in_col.add(box_index(rr,cc))
    claiming_row = int(len(boxes_in_row)==1)
    claiming_col = int(len(boxes_in_col)==1)

    return pointing_row, pointing_col, claiming_row, claiming_col

# --- графовые/окружение ---

def peer_stats(cands: Dict[Tuple[int,int], Set[int]], r:int, c:int) -> Tuple[int,float,float,int,int]:
    """
    Возвращает:
      - n_peers_empty: сколько пустых пиров у (r,c)
      - avg_cand_peers: ср. число кандидатов у пустых пиров
      - max_cand_peers: макс. число кандидатов у пустых пиров
      - peers_with_d: сколько пиров допускает цифру d? (заполним позже на d)
      - peers_singletons: число синглтонов среди пиров
    """
    P = peers_of(r,c)
    empties = [(rr,cc) for (rr,cc) in P if (rr,cc) in cands]
    n = len(empties)
    if n==0: 
        return 0, 0.0, 0.0, 0, 0
    ks = [len(cands[(rr,cc)]) for (rr,cc) in empties]
    peers_singletons = sum(1 for k in ks if k==1)
    return n, float(np.mean(ks)), float(np.max(ks)), 0, peers_singletons

# --- извлечение фич для (r,c,d) ---

def feature_names() -> List[str]:
    names = []

    names += ["k_cand", "is_singleton"]

    for scope in ["row","col","box"]:
        names += [f"{scope}_empties", f"{scope}_singles", f"{scope}_twos", f"{scope}_threes", f"{scope}_ge4",
                  f"{scope}_k_min", f"{scope}_k_avg", f"{scope}_k_max",
                  f"{scope}_entropy"]

    for scope in ["row","col","box"]:
        names += [f"{scope}_places_d", f"{scope}_hidden_d"]

    names += ["pointing_row", "pointing_col", "claiming_row", "claiming_col"]

    for scope in ["row","col","box"]:
        names += [f"{scope}_naked_pair", f"{scope}_naked_triple", f"{scope}_fishy"]

    names += ["peers_empty", "peers_avg_k", "peers_max_k", "peers_with_d", "peers_singletons"]

    names += ["grid_singles","grid_twos","grid_threes","grid_ge4","grid_entropy"]

    names += ["is_row_edge","is_col_edge","is_center_box"]

    return names

def extract_features_for_cell_digit(grid: np.ndarray, r: int, c: int, d: int) -> np.ndarray:
    """
    Возвращает вектор фич (np.float32) для клетки (r,c) и цифры d.
    grid: (9,9) np.int
    """
    assert 0 <= r < 9 and 0 <= c < 9 and 1 <= d <= 9
    C = all_candidates(grid)
    cand_rc = C.get((r,c), set())
    k = len(cand_rc) if grid[r,c]==0 else 0

    feats = []

    feats += [k, 1.0 if k==1 else 0.0]

    row_cells = unit_cells_row(r)
    col_cells = unit_cells_col(c)
    box_cells = unit_cells_box(box_index(r,c))

    def unit_stats(cells):
        clist = [C[(rr,cc)] for (rr,cc) in cells if (rr,cc) in C]
        empties = len(clist)
        ones, twos, threes, ge4 = count_by_size(clist)
        ks = [len(S) for S in clist]
        k_min = float(min(ks)) if ks else 0.0
        k_avg = float(np.mean(ks)) if ks else 0.0
        k_max = float(max(ks)) if ks else 0.0
        counts = [ones, twos, threes, ge4]
        ent = entropy_from_counts(counts)
        return empties, ones, twos, threes, ge4, k_min, k_avg, k_max, ent, clist

    r_stats = unit_stats(row_cells)
    c_stats = unit_stats(col_cells)
    b_stats = unit_stats(box_cells)

    for stats in [r_stats, c_stats, b_stats]:
        empties, ones, twos, threes, ge4, k_min, k_avg, k_max, ent, _ = stats
        feats += [empties, ones, twos, threes, ge4, k_min, k_avg, k_max, ent]

    def places_d(clist, d):
        return sum(1 for S in clist if d in S)

    row_places_d = places_d(r_stats[-1], d)
    col_places_d = places_d(c_stats[-1], d)
    box_places_d = places_d(b_stats[-1], d)

    feats += [row_places_d, 1.0 if row_places_d==1 else 0.0]
    feats += [col_places_d, 1.0 if col_places_d==1 else 0.0]
    feats += [box_places_d, 1.0 if box_places_d==1 else 0.0]

    pr, pc, cr, cc_ = pointing_claiming_flags(C, d, r, c)
    feats += [float(pr), float(pc), float(cr), float(cc_)]

    for stats in [r_stats, c_stats, b_stats]:
        _, _, _, _, _, _, _, _, _, clist = stats
        has_pair, has_triple, fishy = naked_pair_triple_flags(clist)
        feats += [float(has_pair), float(has_triple), float(fishy)]

    n_peers, avg_k, max_k, _, peers_single = peer_stats(C, r, c)
    peers = peers_of(r,c)
    peers_with_d = sum(1 for (rr,cc) in peers if d in C.get((rr,cc), set()))
    feats += [float(n_peers), float(avg_k), float(max_k), float(peers_with_d), float(peers_single)]

    clist_all = list(C.values())
    ones, twos, threes, ge4 = count_by_size(clist_all)
    grid_ent = entropy_from_counts([ones, twos, threes, ge4])
    feats += [float(ones), float(twos), float(threes), float(ge4), float(grid_ent)]

    is_row_edge = int(r in [0,8])
    is_col_edge = int(c in [0,8])
    br, bc = 3*(r//3), 3*(c//3)
    is_center_box = int((br==3) and (bc==3))
    feats += [float(is_row_edge), float(is_col_edge), float(is_center_box)]

    return np.array(feats, dtype=np.float32)


def build_action_feature_matrix(grid: np.ndarray):
    
    assert isinstance(grid, np.ndarray) and grid.shape == (9, 9)
    idxs = []
    rows = []

    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                continue
            C = candidates_for_cell(grid, r, c)
            for d in sorted(C):
                idx = (r * 9 + c) * 9 + (d - 1)
                idxs.append(idx)
                rows.append(extract_features_for_cell_digit(grid, r, c, d))

    if not rows:
        names = feature_names()
        return [], np.zeros((0, len(names)), dtype=np.float32), names

    X = np.vstack(rows).astype(np.float32)
    names = feature_names()
    return idxs, X, names

build_action_features = build_action_feature_matrix
action_feature_matrix = build_action_feature_matrix
compute_action_features = build_action_feature_matrix

