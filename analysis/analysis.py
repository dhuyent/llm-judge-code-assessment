"""
Phan tich toan bo pipeline Stage 0 -> Stage 3 cho dataset v2.

Chay:  python analysis/analysis.py
Xuat:  analysis/output/*.csv  + bao cao text ra stdout
"""
import os, sys, glob, itertools
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "analysis", "output")
os.makedirs(OUT, exist_ok=True)

VARIANTS = ["logic1", "logic2", "reference1"]          # reference2 bi loai sau Stage 0.1
ZS_MODELS = ["qwen3.7-max", "llama-3.3-70b-versatile"]
ESC_MODELS = ["claude-opus-4-8", "gpt-5.6-sol"]
LABEL_NAME = {1: "1_dung", 2: "2_ban_phan", 3: "3_sai"}


def rd(rel, **kw):
    p = os.path.join(ROOT, rel)
    return pd.read_csv(p, **kw) if rel.endswith(".csv") else pd.read_excel(p, **kw)


def norm(s):
    return s.astype(str).str.strip()


def status_ok(df):
    """Dung/sai theo verdict (status-exact match)."""
    return norm(df.gt_status) == norm(df.pred_status)


def dump(df, name):
    path = os.path.join(OUT, name + ".csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[saved] analysis/out/{name}.csv")
    return df


def header(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- B1. Inventory
header("B1. INVENTORY & PHAN BO GROUND TRUTH (Stage 0.1)")
rows = []
for v in VARIANTS + ["reference2"]:
    d = rd(rf"data\stage0\step1\step1_{v}.csv")
    vc = norm(d.gt_status).value_counts()
    rows.append(dict(variant=v, n=len(d), n_id_unique=d.id.nunique(),
                     **{f"gt_{k}": int(x) for k, x in vc.items()}))
inv = pd.DataFrame(rows).fillna(0)
print(inv.to_string(index=False))
dump(inv, "b1_inventory_ground_truth")

meta = rd(r"data\stage0\step1\step1_logic1.csv")
print("\nPhan bo topic:", dict(norm(meta.topic).value_counts()))
print("Phan bo level:", dict(norm(meta.level).value_counts()))


# ------------------------------------------------- B2. Stage 0.2 zero-shot base
header("B2. STAGE 0.2 - ZERO-SHOT BASELINE (status-exact, 3 run/model)")
recs = []
zs_runs = {}
for v in VARIANTS:
    for m in ZS_MODELS:
        for r in (1, 2, 3):
            d = rd(rf"data\stage0\step2\step2_{v}_{m}_{r}.csv").set_index("id")
            ok = status_ok(d)
            zs_runs[(v, m, r)] = ok
            recs.append(dict(variant=v, model=m, run=r, n=len(d),
                             acc=round(ok.mean(), 4), n_wrong=int((~ok).sum())))
zs = pd.DataFrame(recs)
piv = zs.pivot_table(index=["variant", "model"], values="acc", aggfunc=["mean", "std", "min", "max"])
piv.columns = ["acc_mean", "acc_std", "acc_min", "acc_max"]
print(piv.round(4).to_string())
dump(zs, "b2_stage0_zeroshot_per_run")

print("\n-- Self-consistency: % case cho cung ket qua dung/sai o ca 3 run")
cons = []
for v in VARIANTS:
    for m in ZS_MODELS:
        M = pd.concat([zs_runs[(v, m, r)] for r in (1, 2, 3)], axis=1)
        cons.append(dict(variant=v, model=m,
                         stable=round((M.all(axis=1) | (~M).all(axis=1)).mean(), 4),
                         always_right=round(M.all(axis=1).mean(), 4),
                         always_wrong=round((~M).all(axis=1).mean(), 4),
                         flaky=round((M.any(axis=1) & ~M.all(axis=1)).mean(), 4)))
consdf = pd.DataFrame(cons)
print(consdf.to_string(index=False))
dump(consdf, "b2_stage0_self_consistency")


# ------------------------------------------- B3. Label adjudication (classify1)
header("B3. STAGE 1.1 - LABEL THAM DINH & FILTER D1")
c1 = {v: rd(rf"data\stage1\step1\step1_classify1_{v}.csv") for v in VARIANTS}
d1 = {v: rd(rf"data\stage1\step1\step1_filter1_{v}.csv") for v in VARIANTS}
rows = []
for v in VARIANTS:
    c, f = c1[v], d1[v]
    lab = c.label.value_counts()
    fl = c.filtered_label.value_counts()
    expect = set(c[c.filtered_label.isin([2, 3])].id)
    got = set(f.id)
    rows.append(dict(variant=v, n=len(c),
                     label1=int(lab.get(1, 0)), label2=int(lab.get(2, 0)), label3=int(lab.get(3, 0)),
                     flab1=int(fl.get(1, 0)), flab2=int(fl.get(2, 0)), flab3=int(fl.get(3, 0)),
                     acc_strict=round((c.label == 1).mean(), 4),
                     acc_adjudicated=round((c.filtered_label == 1).mean(), 4),
                     D1_size=len(f),
                     D1_khop_flabel23=(expect == got),
                     ro_ri=len(got - expect)))
b3 = pd.DataFrame(rows)
print(b3.to_string(index=False))
dump(b3, "b3_stage1_classify_filter")

print("\n-- Chenh lech label (theo status) vs filtered_label (tham dinh sau):")
for v in VARIANTS:
    c = c1[v]
    ct = pd.crosstab(c.label, c.filtered_label)
    print(f"\n[{v}] hang=label, cot=filtered_label\n{ct.to_string()}")
    bad = c[(c.label == 1) & (c.filtered_label != 1)]
    print(f"  -> {len(bad)} case 'dung verdict nhung sai lap luan/testcase' bi ha bac.")


# ---------------------------------------------------- B4. Stage 1.2 / 1.3
header("B4. STAGE 1.2 (Qwen few-shot tren D1) & 1.3 (Claude/GPT zero-shot tren D1)")
recs = []
for v in VARIANTS:
    for r in (1, 2, 3):
        d = rd(rf"data\stage1\step2\step2_filter1_{v}_qwen3.7-max_{r}.csv")
        ok = status_ok(d)
        recs.append(dict(variant=v, stage="1.2", model="qwen3.7-max", strategy="few-shot",
                         run=r, n=len(d), acc=round(ok.mean(), 4),
                         strategy_col=dict(norm(d.prompt_strategy).value_counts())))
    for m in ESC_MODELS:
        for r in (1, 2):
            d = rd(rf"data\stage1\step3\step3_filter1_{v}_{m}_{r}.csv")
            ok = status_ok(d)
            recs.append(dict(variant=v, stage="1.3", model=m, strategy="zero-shot",
                             run=r, n=len(d), acc=round(ok.mean(), 4),
                             strategy_col=dict(norm(d.prompt_strategy).value_counts())))
b4 = pd.DataFrame(recs)
print(b4.drop(columns="strategy_col").to_string(index=False))
print("\n-- Canh bao cot prompt_strategy khong dong nhat:")
for _, r in b4.iterrows():
    if len(r.strategy_col) > 1:
        print(f"   {r.variant} {r.stage} {r.model} run{r.run}: {r.strategy_col}")
dump(b4.drop(columns="strategy_col"), "b4_stage1_fewshot_and_escalation")

print("\n-- Recovery rate tren D1 (D1 = 100% sai o Stage 0):")
rec = (b4.groupby(["variant", "stage", "model"]).acc.agg(["mean", "std", "count"])
         .round(4).rename(columns={"mean": "recovery_mean", "std": "recovery_std", "count": "n_run"}))
print(rec.to_string())
dump(rec.reset_index(), "b4_recovery_rate_D1")


# ------------------------------------------------------------ B5. Stage 2
header("B5. STAGE 2.1 (filter D2) & 2.2 (few-shot Claude/GPT tren D2)")
rows = []
for v in VARIANTS:
    cc = rd(rf"data\stage2\step1\step1_classify2_{v}_claude-opus-4-8.csv")
    cg = rd(rf"data\stage2\step1\step1_classify2_{v}_gpt-5.6-sol.csv")
    f2 = rd(rf"data\stage2\step1\step1_filter2_{v}.csv")
    C = set(cc[cc.filtered_label.isin([2, 3])].id)
    G = set(cg[cg.filtered_label.isin([2, 3])].id)
    qfs_wrong = set.intersection(*[set(rd(rf"data\stage1\step2\step2_filter1_{v}_qwen3.7-max_{r}.csv")
                                       .pipe(lambda d: d[~status_ok(d)]).id) for r in (1, 2, 3)])
    rows.append(dict(variant=v, D1=len(cc),
                     claude_zs_acc=round((cc.filtered_label == 1).mean(), 4),
                     gpt_zs_acc=round((cg.filtered_label == 1).mean(), 4),
                     claude_wrong=len(C), gpt_wrong=len(G),
                     D2=len(f2),
                     D2_eq_claude_AND_gpt=(C & G == set(f2.id)),
                     D2_eq_claude_AND_qwenFS=((C & qfs_wrong) == set(f2.id)),
                     qwen_fs_wrong_all3=len(qfs_wrong),
                     D2_subset_qwenFSwrong=set(f2.id) <= qfs_wrong))
b5 = pd.DataFrame(rows)
print(b5.to_string(index=False))
dump(b5, "b5_stage2_filter")

recs = []
for v in VARIANTS:
    for m in ESC_MODELS:
        for r in (1, 2):
            d = rd(rf"data\stage2\step2\step2_filter2_{v}_{m}_{r}.csv")
            recs.append(dict(variant=v, model=m, strategy="few-shot", run=r,
                             n=len(d), acc=round(status_ok(d).mean(), 4)))
b5b = pd.DataFrame(recs)
print("\n-- Stage 2.2 few-shot tren D2 (status-exact):")
print(b5b.to_string(index=False))
dump(b5b, "b5_stage2_fewshot")


# ------------------------------------------------------------ B6. Stage 3
header("B6. STAGE 3 - TONG HOP")
print("\n[3.1] BANG ATTRITION")
att = []
for v in VARIANTS:
    n0 = len(c1[v]); n1 = len(d1[v]); n2 = len(rd(rf"data\stage2\step1\step1_filter2_{v}.csv"))
    c3c = rd(rf"data\stage3\classify3_{v}_claude-opus-4-8.csv")
    c3g = rd(rf"data\stage3\classify3_{v}_gpt-5.6-sol.csv")
    resid_c = int((c3c.label != 1).sum()); resid_g = int((c3g.label != 1).sum())
    resid_both = len(set(c3c[c3c.label != 1].id) & set(c3g[c3g.label != 1].id))
    att.append(dict(variant=v, D0=n0, D1=n1, D2=n2,
                    pct_D1=round(n1 / n0, 4), pct_D2=round(n2 / n0, 4),
                    solved_S1=n0 - n1, solved_S2=n1 - n2,
                    residual_claudeFS=resid_c, residual_gptFS=resid_g,
                    residual_ca_hai=resid_both,
                    acc_cuoi_claudeFS=round((c3c.label == 1).mean(), 4),
                    acc_cuoi_gptFS=round((c3g.label == 1).mean(), 4)))
attdf = pd.DataFrame(att)
print(attdf.to_string(index=False))
dump(attdf, "b6_attrition")

print("\n[3.2] ACCURACY TOAN PIPELINE (cumulative, tren D0 = 179/nhanh)")
cum = []
for _, r in attdf.iterrows():
    n0 = r.D0
    cum.append(dict(variant=r.variant,
                    acc_S0_qwen_zs=round((n0 - r.D1) / n0, 4),
                    acc_sau_S1=round((n0 - r.D2) / n0, 4),
                    acc_cuoi_pipeline=round((n0 - r.residual_ca_hai) / n0, 4),
                    con_lai_khong_giai_duoc=r.residual_ca_hai))
cumdf = pd.DataFrame(cum)
print(cumdf.to_string(index=False))
dump(cumdf, "b6_accuracy_pipeline")

print("\n[3.3] PHAN TICH THEO VERDICT / TOPIC / LEVEL")
long = []
for v in VARIANTS:
    c = c1[v][["id", "topic", "level", "gt_status", "label", "filtered_label"]].copy()
    c["variant"] = v
    c["gt_status"] = norm(c.gt_status)
    c["in_D1"] = c.id.isin(set(d1[v].id))
    d2ids = set(rd(rf"data\stage2\step1\step1_filter2_{v}.csv").id)
    c["in_D2"] = c.id.isin(d2ids)
    c3c = rd(rf"data\stage3\classify3_{v}_claude-opus-4-8.csv").set_index("id")
    c3g = rd(rf"data\stage3\classify3_{v}_gpt-5.6-sol.csv").set_index("id")
    resid = set(c3c[c3c.label != 1].index) & set(c3g[c3g.label != 1].index)
    c["residual"] = c.id.isin(resid)
    long.append(c)
L = pd.concat(long, ignore_index=True)
dump(L, "b6_case_level_long")

for dim in ["gt_status", "topic", "level"]:
    g = L.groupby(dim).agg(n=("id", "size"),
                           acc_S0=("in_D1", lambda s: round(1 - s.mean(), 4)),
                           rate_D1=("in_D1", lambda s: round(s.mean(), 4)),
                           rate_D2=("in_D2", lambda s: round(s.mean(), 4)),
                           rate_residual=("residual", lambda s: round(s.mean(), 4)))
    g = g.sort_values("n", ascending=False)
    print(f"\n-- Theo {dim} (gop 3 nhanh, N={len(L)}):\n{g.to_string()}")
    dump(g.reset_index(), f"b6_by_{dim}")

print("\n-- Confusion Stage 0 (gt_status x pred_status, qwen zero-shot run1, gop 3 nhanh):")
cm = []
for v in VARIANTS:
    d = rd(rf"data\stage0\step2\step2_{v}_qwen3.7-max_1.csv")
    cm.append(pd.DataFrame({"gt": norm(d.gt_status), "pred": norm(d.pred_status)}))
CMd = pd.concat(cm, ignore_index=True)
CM = pd.crosstab(CMd["gt"], CMd["pred"])
print(CM.to_string())
dump(CM.reset_index(), "b6_confusion_stage0_qwen")

print("\n\nHoan tat. Tat ca bang nam trong analysis/output/")