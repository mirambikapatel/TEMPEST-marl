# Run from marl_multiseed/ :  python3 pheme_compare_v2.py
"""Like-for-like comparison of simulated vs real PHEME cascades.
Real: cumulative sharers (source authors + 'agreed') and cumulative refuters ('disagreed'),
      binned on the same 60-step clock as the simulation (meta.json t_min, dt_sec).
Sim:  cumulative ever-reached  E(t) = N - S(t)   and  cumulative refuters R_refuted(t).
Both normalised by their own final value -> shape comparison.
Veracity filter from each thread's annotation.json (true / false / unverified)."""
import json, glob, datetime as dt, numpy as np, os

RAW='OUR_program/pheme/raw/pheme-rumour-scheme-dataset'; RES='results_v2/pheme'; T=60
ann={}
for line in open(f'{RAW}/annotations/en-scheme-annotations.json'):
    line=line.strip()
    if not line or line.startswith('#'): continue
    o=json.loads(line); ann[str(o['tweetid'])]=o
def ts(f):
    d=json.load(open(f)); return str(d['user']['id']), dt.datetime.strptime(d['created_at'],'%a %b %d %H:%M:%S %z %Y')
def veracity(tdir):
    a=json.load(open(tdir+'/annotation.json'))
    return 'true' if str(a.get('true'))=='1' else ('false' if str(a.get('misinformation'))=='1' else 'unverified')
def real_curves(ev, meta, keep):
    t0=dt.datetime.fromisoformat(meta['t_min_iso']); dts=meta['dt_sec']
    S,Rf={}, {}
    for tdir in sorted(glob.glob(f'{RAW}/threads/en/{ev}/*')):
        if veracity(tdir) not in keep: continue
        for f in glob.glob(tdir+'/source-tweets/*.json'):
            u,t=ts(f); S[u]=min(S.get(u,t),t)
        for f in glob.glob(tdir+'/reactions/*.json'):
            r=ann.get(os.path.basename(f)[:-5],{}).get('responsetype-vs-source','')
            if r not in ('agreed','disagreed'): continue
            u,t=ts(f); D=S if r=='agreed' else Rf; D[u]=min(D.get(u,t),t)
    def cum(D):
        c=np.zeros(T)
        for t in D.values():
            k=min(T-1,max(0,int((t-t0).total_seconds()//dts))); c[k:]+=1
        return c
    return cum(S), cum(Rf)
def shape(c): return c/c[-1] if c[-1]>0 else c*np.nan
def t_frac(c,f): s=shape(c); return int(np.argmax(s>=f)) if np.all(np.isfinite(s)) else np.nan
def cmp(sim, real):
    a,b=shape(sim),shape(real)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))): return dict(r=np.nan,rmse=np.nan,dt50=np.nan)
    return dict(r=np.corrcoef(a,b)[0,1], rmse=np.sqrt(np.mean((a-b)**2)), dt50=t_frac(sim,.5)-t_frac(real,.5))
EV=['charliehebdo','ottawashooting','sydneysiege','ferguson','germanwings-crash']
out={}
for keep_name, keep in [('all',{'true','false','unverified'}),('false+unverified',{'false','unverified'})]:
    print(f'\n######## threads: {keep_name}')
    for ev in EV:
        meta=json.load(open(f'{RES}/{ev}/meta.json')); N=meta['n_nodes']
        rs, rr = real_curves(ev, meta, keep)
        J=[json.load(open(f)) for f in sorted(glob.glob(f'{RES}/{ev}/seed_*.json'))]
        print(f"\n{ev}: real sharers {int(rs[-1])}, refuters {int(rr[-1])}, refuters/(sharers+refuters) {rr[-1]/(rs[-1]+rr[-1]):.2f}; real t50 sharers {t_frac(rs,.5)}, refuters {t_frac(rr,.5)}")
        for m in ['tempest','type_prior','sir']:
            res=[]
            for j in J:
                c=j['curves'][m]; E=N-np.array(c['S'],float); Rr=np.array(c['R_refuted'],float); Rall=np.array(c['R'],float)
                cs=cmp(E,rs); cr=cmp(Rr,rr) if Rr[-1]>0 else dict(r=np.nan,rmse=np.nan,dt50=np.nan)
                res.append([cs['r'],cs['rmse'],cs['dt50'],cr['r'],cr['rmse'],cr['dt50'],Rr[-1]/E[-1], E[-1]/N, t_frac(E,.5)])
            A=np.array(res,float); mu=np.nanmean(A,0); se=np.nanstd(A,0,ddof=1)/np.sqrt(len(A))
            print(f"  {m:10s} share-shape r={mu[0]:.2f}±{se[0]:.2f} rmse={mu[1]:.3f} Δt50={mu[2]:+.1f} | refute-shape r={mu[3]:.2f} rmse={mu[4]:.3f} Δt50={mu[5]:+.1f} | refuted/reached={mu[6]:.2f} reach={mu[7]:.2f} sim_t50={mu[8]:.1f}")
            out[(keep_name,ev,m)]=mu.tolist()
json.dump({'|'.join(k):v for k,v in out.items()},open(f'{RES}/compare_v2.json','w'),indent=1)
