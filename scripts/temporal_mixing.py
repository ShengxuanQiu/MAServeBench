"""Predeclare and audit a fixed-population background trace-to-slot comparison.

Input is an existing S65 heterogeneous block plus its immutable trace bank.
No serving is started by this CPU-only program. Stable round-robin class order
and class-clustered order are chosen before outcomes exist; no seed search.
"""
import argparse, collections, copy, hashlib, json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def population(roots, bank):
    rows=[]
    for root in roots:
        key=root["binding"]+"/"+root["workload"]
        ops=bank[key]["operations"]
        rows.append({"root":{k:v for k,v in root.items() if k!="arrival"},
                     "trace_key":key,"operations_hash":digest(ops)})
    return sorted(rows,key=lambda x:x["root"]["root_id"])


def audit(left, right, bank):
    assert len({r["root_id"] for r in left})==len(left),"Duplicate root IDs"
    assert len({r["root_id"] for r in right})==len(right),"Duplicate root IDs"
    assert population(left,bank)==population(right,bank),"Root/trace population changed"
    targets=lambda rs:sorted((r for r in rs if r.get("target")),key=lambda r:r["root_id"])
    assert targets(left)==targets(right),"Target identity or arrivals changed"
    slots=lambda rs:sorted(r["arrival"] for r in rs if not r.get("target"))
    assert slots(left)==slots(right),"Global background arrival slots changed"
    totals=[]
    for rs in [left,right]:
        llms=[op["llm"] for r in rs for op in bank[r["binding"]+"/"+r["workload"]]["operations"] if op.get("llm")]
        totals.append(dict(calls=len(llms),input_tokens=sum(x["input_token_count"] for x in llms),
                           output_tokens=sum(x["recorded_output_length"] for x in llms)))
    assert totals[0]==totals[1],"Model work changed"
    return dict(passed=True,population_hash=digest(population(left,bank)),
                background_slots_hash=digest(slots(left)),targets_hash=digest(targets(left)),
                classes=dict(collections.Counter(r["workload"] for r in left if not r.get("target"))),
                targets=len(targets(left)),background=len(slots(left)),totals=totals[0])


def make_block(roots,bank):
    roots=copy.deepcopy(roots)
    targets=[r for r in roots if r.get("target")]
    bg=[r for r in roots if not r.get("target")]
    slots=sorted(r["arrival"] for r in bg)
    groups={c:sorted([r for r in bg if r["workload"]==c],key=lambda r:r["root_id"])
            for c in sorted({r["workload"] for r in bg})}
    clustered=[r for c in groups for r in groups[c]]
    interleaved=[]
    for i in range(max(map(len,groups.values()))):
        interleaved.extend(groups[c][i] for c in groups if i<len(groups[c]))
    arms={}
    for name,order in [("interleaved",interleaved),("clustered",clustered)]:
        mapped=[dict(r,arrival=t) for r,t in zip(order,slots)]
        arms[name]=sorted(copy.deepcopy(targets)+mapped,key=lambda r:(r["arrival"],r["root_id"]))
    return dict(**arms,audit=audit(arms["interleaved"],arms["clustered"],bank))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--matching",required=True)
    p.add_argument("--bank",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    match=json.loads(Path(a.matching).read_text());bank=json.loads(Path(a.bank).read_text())
    blocks=[dict(block=m["block"],**make_block(m["heterogeneous"],bank)) for m in match["blocks"]]
    plan=dict(protocol="fixed_population_temporal_mixing_v1",blocks=blocks,
              ordering="Lexicographic class order, stable root-ID order within class; round-robin interleaving versus class concatenation",
              primary="Mean and per-target paired latency, then completion-path accounting; three blocks, all outcomes",
              scope="Changes temporal overlap of classes; individual background arrivals change; target arrivals and global slots fixed")
    plan["manifest_sha256"]=digest(plan)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:json.dump(plan,f,indent=2)
    print(json.dumps({"manifest_sha256":plan["manifest_sha256"],"audit":[b["audit"] for b in blocks]}))


if __name__=="__main__":main()
