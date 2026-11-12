"""
pheme_prepare.py — build PHEME event networks and real cascade curves.

Reads: OUR_program/pheme/raw/pheme-rumour-scheme-dataset/
Writes:
  results_v2/pheme/{event}/network.gpickle       (networkx Graph, LCC)
  results_v2/pheme/{event}/real_cascade.csv      (timestamped I/R curves)
  results_v2/pheme/{event}/meta.json             (N, E, K, init_infected user ids)

Usage:
  python3 pheme_prepare.py                       # all qualified events
  python3 pheme_prepare.py --events charliehebdo # specific events
  python3 pheme_prepare.py --smoke               # first thread only per event
"""
import argparse
import csv
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
PHEME_RAW = HERE / 'OUR_program' / 'pheme' / 'raw' / 'pheme-rumour-scheme-dataset'
OUT_BASE = HERE / 'results_v2' / 'pheme'

LCC_MIN_NODES = 1000   # events with LCC < this are excluded
MAX_TIME = 60          # simulation steps (matches revision_runner.py default)
TWITTER_FMT = '%a %b %d %H:%M:%S +0000 %Y'

ENGLISH_EVENTS = [
    'charliehebdo', 'ottawashooting', 'sydneysiege',
    'ferguson', 'germanwings-crash', 'putinmissing', 'prince-toronto', 'ebola-essien'
]


def load_annotations(lang='en'):
    """Return two dicts: src_ann[tweetid] = support, react_ann[tweetid] = responsetype-vs-source."""
    path = PHEME_RAW / 'annotations' / f'{lang}-scheme-annotations.json'
    src_ann, react_ann = {}, {}
    for line in path.read_text(errors='replace').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = o.get('tweetid', '')
        thr = o.get('threadid', '')
        if tid == thr:
            src_ann[tid] = o.get('support', '')
        else:
            rtype = o.get('responsetype-vs-source', '')
            if rtype:
                react_ann[tid] = rtype
    return src_ann, react_ann


def parse_twitter_ts(ts_str):
    """Parse Twitter created_at string → UTC datetime."""
    return datetime.strptime(ts_str, TWITTER_FMT).replace(tzinfo=timezone.utc)


def load_tweet_ts(json_path):
    """Return (user_id, created_at_datetime) from a tweet JSON file."""
    try:
        d = json.loads(Path(json_path).read_text(errors='replace'))
        uid = str(d['user']['id'])
        ts = parse_twitter_ts(d['created_at'])
        return uid, ts
    except Exception:
        return None, None


def build_event(event, threads_dir, src_ann, react_ann, smoke=False):
    """
    Build network and cascade for one event.

    Returns dict with keys: G (LCC), tweets (list of (uid, ts, role)),
    init_uids (users who are earliest sharers in the network), meta.
    """
    event_dir = threads_dir / event
    threads = sorted(event_dir.iterdir())
    if smoke:
        threads = threads[:1]

    # ── 1. Network ────────────────────────────────────────────────────────────
    DG = nx.DiGraph()
    for tdir in threads:
        wfw = tdir / 'who-follows-whom.dat'
        if wfw.exists():
            for line in wfw.read_text(errors='replace').splitlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    DG.add_edge(parts[0], parts[1])  # follower → followed

    # Undirected, relabelled 0..N-1 for TEMPEST
    UG = DG.to_undirected()
    UG.remove_edges_from(nx.selfloop_edges(UG))
    if UG.number_of_nodes() == 0:
        return None

    lcc_nodes = max(nx.connected_components(UG), key=len)
    G = nx.convert_node_labels_to_integers(
        UG.subgraph(lcc_nodes).copy(), label_attribute='orig_id')
    orig_to_int = {G.nodes[v]['orig_id']: v for v in G.nodes}

    # ── 2. Cascade tweets ─────────────────────────────────────────────────────
    tweets = []   # (user_id_str, datetime, role)  role: 'share'|'refute'|'neutral'
    for tdir in threads:
        src_dir = tdir / 'source-tweets'
        if src_dir.exists():
            for f in src_dir.glob('*.json'):
                uid, ts = load_tweet_ts(f)
                if uid and ts:
                    role = 'share'   # source tweet always counts as sharing
                    tweets.append((uid, ts, role))
        react_dir = tdir / 'reactions'
        if react_dir.exists():
            for f in react_dir.glob('*.json'):
                uid, ts = load_tweet_ts(f)
                if uid and ts:
                    rtype = react_ann.get(str(f.stem), '')
                    if rtype == 'agreed':
                        role = 'share'
                    elif rtype == 'disagreed':
                        role = 'refute'
                    else:
                        role = 'neutral'
                    tweets.append((uid, ts, role))

    if not tweets:
        return None

    tweets.sort(key=lambda x: x[1])
    t_min = tweets[0][1]
    t_max = tweets[-1][1]
    span = max((t_max - t_min).total_seconds(), 1.0)
    dt_sec = span / (MAX_TIME - 1)

    def to_sim_t(ts):
        return min(MAX_TIME - 1, int((ts - t_min).total_seconds() / dt_sec))

    # ── 3. Cascade curves ─────────────────────────────────────────────────────
    I_at = defaultdict(set)   # sim_t → set of user_ids (sharers)
    R_at = defaultdict(set)   # sim_t → set of user_ids (refuters)
    for uid, ts, role in tweets:
        t = to_sim_t(ts)
        if role == 'share':
            I_at[t].add(uid)
        elif role == 'refute':
            R_at[t].add(uid)

    # Cumulative
    I_cum, R_cum = set(), set()
    I_curve, R_curve = [], []
    for t in range(MAX_TIME):
        I_cum.update(I_at.get(t, set()))
        R_cum.update(R_at.get(t, set()))
        I_curve.append(len(I_cum))
        R_curve.append(len(R_cum))

    total_participants = max(len(I_cum) + len(R_cum), 1)
    I_norm = [x / total_participants for x in I_curve]
    R_norm = [x / total_participants for x in R_curve]

    # ── 4. Initially infected = earliest sharers in the LCC ───────────────────
    # Collect users who share at the earliest simulation step
    earliest_t = min(I_at.keys()) if I_at else 0
    init_uids_str = I_at.get(earliest_t, set())
    # Map to integer node ids (keep only those present in LCC)
    init_nodes = sorted(set(orig_to_int[u] for u in init_uids_str if u in orig_to_int))
    if not init_nodes:
        # Fall back: any sharing user present in LCC
        init_nodes = sorted(set(orig_to_int[u] for u in I_cum if u in orig_to_int))[:1]
    if not init_nodes:
        init_nodes = [0]   # ultimate fallback

    return dict(
        G=G,
        I_curve=I_curve, R_curve=R_curve,
        I_norm=I_norm, R_norm=R_norm,
        total_participants=total_participants,
        init_nodes=init_nodes,
        dt_sec=dt_sec, t_min_iso=t_min.isoformat(),
        n_sharers=len(I_cum), n_refuters=len(R_cum),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    threads_dir = PHEME_RAW / 'threads' / 'en'
    src_ann, react_ann = load_annotations('en')

    events = args.events or ENGLISH_EVENTS
    qualified = []

    for event in events:
        event_dir = threads_dir / event
        if not event_dir.exists():
            print(f'{event}: directory not found, skipping')
            continue
        print(f'\n=== {event} ===')
        result = build_event(event, threads_dir, src_ann, react_ann, smoke=args.smoke)
        if result is None:
            print(f'  SKIP: could not build (no network or no cascade data)')
            continue

        G = result['G']
        N = G.number_of_nodes()
        E = G.number_of_edges()
        print(f'  LCC: N={N}, E={E}')

        if N < LCC_MIN_NODES and not args.smoke:
            print(f'  SKIP: LCC too small (< {LCC_MIN_NODES})')
            continue

        # Community count
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        K = max(1, len(comms))
        print(f'  Communities (greedy modularity): K={K}')
        K_use = min(K, 4)
        if K < 4:
            print(f'  NOTE: K={K} < 4; using all {K} communities (see PHEME_PLAN.md)')

        out_dir = OUT_BASE / event
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save network
        with open(out_dir / 'network.gpickle', 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save cascade CSV
        with open(out_dir / 'real_cascade.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'I_norm', 'R_norm', 'I_count', 'R_count'])
            for t in range(MAX_TIME):
                w.writerow([t, result['I_norm'][t], result['R_norm'][t],
                            result['I_curve'][t], result['R_curve'][t]])

        # Save metadata
        meta = {
            'event': event, 'n_nodes': N, 'n_edges': E,
            'K_communities': K, 'K_use': K_use,
            'init_nodes': result['init_nodes'],
            'n_sharers': result['n_sharers'],
            'n_refuters': result['n_refuters'],
            'total_participants': result['total_participants'],
            'dt_sec': result['dt_sec'],
            't_min_iso': result['t_min_iso'],
            'lcc_min_nodes_threshold': LCC_MIN_NODES,
        }
        (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

        print(f'  Sharers={result["n_sharers"]}, Refuters={result["n_refuters"]}, '
              f'init_nodes={len(result["init_nodes"])}')
        print(f'  Written to {out_dir}')
        qualified.append(event)

    print(f'\nQualified events: {qualified}')
    return qualified


if __name__ == '__main__':
    main()
