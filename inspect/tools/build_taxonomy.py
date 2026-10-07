"""Build steer_bench/taxonomy.json from the site content in autosteer (a derived copy).

usage: python build_taxonomy.py AUTOSTEER_REPO OUT_JSON
"""
import json, subprocess, sys
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
C = repo / "src/autosteer/web/content"
sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
elements = json.loads((C / "elements.json").read_text())
by_name = {}
for e in elements:
    by_name.setdefault((e["version"], e["element"]), e)


def entry(e, eid=None):
    return {"element": e["element"], "name": e.get("display_name") or e["element"], "id": eid,
            "held": e.get("status") == "held"}


res = {"source": (f"Derived copy of autosteer {sha[:7]} src/autosteer/web/content/"
                  "{elements,steer_taxonomy,steer_me_taxonomy}.json (settings, modules and elements in "
                  "paper order, display names, held status). Regenerate with tools/build_taxonomy.py."),
       "benchmarks": {}}
# STEER: elements are listed in the taxonomy in paper order
t = json.loads((C / "steer_taxonomy.json").read_text())
settings = []
for s in t["settings"]:
    mods = []
    for m in s["modules"]:
        els = []
        for x in m.get("elements", []):
            e = by_name.get(("STEER", x["data_name"]))
            if e is None:
                continue  # paper element that is not in the dataset
            els.append(entry(e, x["id"]))
        mods.append({"number": m["number"], "name": m["name"], "elements": els})
    settings.append({"number": s["number"], "name": s["name"], "modules": mods})
res["benchmarks"]["steer"] = {"name": "STEER", "settings": settings}
listed = {x["element"] for s in settings for m in s["modules"] for x in m["elements"]}
missing = [e["element"] for e in elements if e["version"] == "STEER" and e["element"] not in listed]
print("STEER elements not in taxonomy:", missing)

# STEER-ME: settings/modules from the taxonomy, elements from elements.json in file order
t = json.loads((C / "steer_me_taxonomy.json").read_text())
settings = [{"number": s["number"], "name": s["name"],
             "modules": [{"number": m["number"], "name": m["name"], "elements": []} for m in s["modules"]]}
            for s in t["settings"]]
idx = {(s["name"], m["name"]): m for s in settings for m in s["modules"]}
cands = [(e, e["setting"], e["module"]) for e in elements if e["version"] == "STEER-ME"]
for e in elements:
    for a in e.get("also_in") or []:
        if a.get("benchmark") == "STEER-ME":
            cands.append((e, a.get("setting"), a.get("module") or e["module"]))
for e, sname, mname in cands:
    m = idx.get((sname, mname))
    if m is None:
        s = next((s for s in settings if s["name"] == sname), None)
        if s is None:
            print("STEER-ME: no setting", sname, "for", e["element"]); continue
        m = {"number": f"{s['number']}.x", "name": mname, "elements": []}
        s["modules"].append(m); idx[(sname, mname)] = m
        print("STEER-ME: added module", sname, "/", mname)
    m["elements"].append(entry(e))
res["benchmarks"]["steer_me"] = {"name": "STEER-ME", "settings": settings}
for b, v in res["benchmarks"].items():
    print(b, sum(len(m["elements"]) for s in v["settings"] for m in s["modules"]), "elements")
out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
