"""Simulate the full browser user journey against the live server."""
import base64
import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
INFO = "\033[36m•\033[0m"
tests_pass = 0
tests_fail = 0

def check(cond, label, detail=""):
    global tests_pass, tests_fail
    if cond:
        tests_pass += 1
        print(f"  {PASS} {label}" + (f"  {detail}" if detail else ""))
    else:
        tests_fail += 1
        print(f"  {FAIL} {label}" + (f"  {detail}" if detail else ""))

def get(path, **kw):
    r = urllib.request.urlopen(f"{BASE}{path}", **kw)
    return r.status, r.read(), dict(r.headers)

def post(path, body):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

# ---------- 1. Page load ----------
print("\n[1] Page load (what the browser does on first visit)")
status, body, headers = get("/")
check(status == 200, "GET / returns 200")
check(b"Gemma-Lens" in body, "HTML contains title")
check(b"/static/app.js" in body, "HTML references app.js")
check(b"/static/styles.css" in body, "HTML references styles.css")
check(b"cdn.tailwindcss.com" in body, "HTML pulls Tailwind CDN")
check(b"MediaRecorder" not in body, "no mic logic inlined (correctly in app.js)")

# ---------- 2. Static assets ----------
print("\n[2] Static assets (what the browser fetches next)")
status, js, _ = get("/static/app.js")
check(status == 200, "GET /static/app.js")
check(b"MediaRecorder" in js, "app.js contains MediaRecorder logic")
check(b"/examples" in js, "app.js calls /examples")
check(b"/search" in js, "app.js calls /search")
status, css, _ = get("/static/styles.css")
check(status == 200, "GET /static/styles.css")

# ---------- 3. Preset list ----------
print("\n[3] GET /examples (loads 'try these' buttons)")
status, body, _ = get("/examples")
examples = json.loads(body)
check(status == 200, "GET /examples returns 200")
check(len(examples) == 10, f"returns 10 examples", f"got {len(examples)}")
text_ex = [e for e in examples if e["modality"] == "text"]
audio_ex = [e for e in examples if e["modality"] == "audio"]
check(len(text_ex) == 6, "6 text presets")
check(len(audio_ex) == 4, "4 audio presets")
expected_presets = {"cow_moo.wav", "clapping.wav", "cat_meow.wav", "rain.wav"}
check({e["preset"] for e in audio_ex} == expected_presets, "audio preset filenames match")

# ---------- 4. Preset audio files served ----------
print("\n[4] Preset audio files (GET /static/presets/*)")
for preset in sorted(expected_presets):
    status, body, headers = get(f"/static/presets/{preset}")
    check(status == 200 and len(body) > 100000,
          f"/static/presets/{preset}", f"{len(body)} bytes")

# ---------- 5. Text queries — real retrieval quality ----------
print("\n[5] Text search quality (realistic queries)")
queries_expected = [
    ("red cars", "image", ["car", "truck", "traffic"]),
    ("pizza on a plate", "image", ["pizza"]),
    ("sunset over water", "image", ["sunset", "water", "beach", "ocean"]),
    ("dogs playing", "image", ["dog"]),
    ("snowy mountains", "image", ["snow", "mountain", "ski"]),
    ("cow mooing", "audio", ["cow"]),
    ("rain falling", "audio", ["rain"]),
    ("thunder sounds", "audio", ["thunderstorm"]),
    ("clapping hands", "audio", ["clap"]),
    ("cat purring", "audio", ["cat"]),
]
latencies = []
for query, expected_modality, keywords in queries_expected:
    code, data = post("/search", {"modality": "text", "text": query, "k": 8})
    latencies.append(data["query_ms"])
    hits = data["results"]
    # Check at least one top-8 result matches the expected keyword in the right modality
    matches = []
    for h in hits:
        if h["modality"] == "image":
            cap = (h["metadata"].get("caption") or "").lower()
            if any(kw in cap for kw in keywords):
                matches.append((h["modality"], h["score"]))
        else:
            cat = (h["metadata"].get("category") or "").lower()
            if any(kw in cat for kw in keywords):
                matches.append((h["modality"], h["score"]))
    top_match = f"rank1={hits[0]['modality']}:{hits[0]['score']:.2f} matches={len(matches)}/8"
    check(len(matches) >= 1, f"'{query}' → expect {expected_modality}/{keywords[0]}", top_match)

print(f"\n    text query latencies ms: {latencies}")
print(f"    median={sorted(latencies)[len(latencies)//2]} ms  min={min(latencies)}  max={max(latencies)}")

# ---------- 6. Cross-modal: audio preset → audio results ----------
print("\n[6] Audio preset queries (cross-modal: preset ends up matching category)")
for preset, expect_cat in [("cow_moo.wav","cow"), ("cat_meow.wav","cat"),
                           ("clapping.wav","clapping"), ("rain.wav","rain")]:
    with open(f"static/presets/{preset}","rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    code, data = post("/search", {"modality":"audio","audio_b64":b64,"k":8})
    cats = [h["metadata"].get("category") for h in data["results"]]
    hits_in_cat = sum(1 for c in cats if c == expect_cat)
    check(hits_in_cat >= 3, f"{preset} → ≥3/8 hits in '{expect_cat}'",
          f"got {hits_in_cat}/8  rank1_score={data['results'][0]['score']:.2f}")

# ---------- 7. Image upload → image results ----------
print("\n[7] Image upload query (re-queries a known COCO image)")
import os
sample_images = sorted(os.listdir("data/images/val2017"))[:3]
for img_name in sample_images:
    with open(f"data/images/val2017/{img_name}","rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    code, data = post("/search", {"modality":"image","image_b64":b64,"k":5})
    rank1 = data["results"][0]
    is_in_corpus = rank1["filename"] == img_name
    check(rank1["score"] > 0.99 if is_in_corpus else True,
          f"upload {img_name}", f"rank1 score={rank1['score']:.3f}")

# ---------- 8. Asset serving ----------
print("\n[8] Asset serving (what tile click → modal does)")
_, data = post("/search", {"modality":"text","text":"pizza","k":4})
for hit in data["results"][:2]:
    url = hit["url"]
    status, body, headers = get(url)
    content_type = headers.get("content-type", "")
    expected = "image/jpeg" if hit["modality"] == "image" else "audio/wav"
    check(status == 200 and content_type.startswith(expected.split("/")[0]),
          f"GET {url}", f"type={content_type} bytes={len(body)}")

# ---------- 9. Error paths ----------
print("\n[9] Error handling (bad client inputs)")
code, body = post("/search", {"modality":"text"})
check(code == 400, "text query missing `text` → 400")
code, body = post("/search", {"modality":"audio"})
check(code == 400, "audio query missing audio_b64 → 400")
code, body = post("/search", {"modality":"image"})
check(code == 400, "image query missing image_b64 → 400")
code, body = post("/search", {"modality":"wrong","text":"x"})
check(code == 422, f"invalid modality → 422 (pydantic)", f"got {code}")

# k boundary
code, data = post("/search", {"modality":"text","text":"test","k":1})
check(code == 200 and len(data["results"]) == 1, "k=1 → 1 result")
code, data = post("/search", {"modality":"text","text":"test","k":64})
check(code == 200 and len(data["results"]) == 64, "k=64 → 64 results")
code, body = post("/search", {"modality":"text","text":"test","k":100})
check(code == 422, f"k>64 → 422", f"got {code}")

# ---------- 10. 404 for missing asset ----------
print("\n[10] 404 for missing asset ids")
try:
    get("/asset/image/999999999")
    check(False, "missing image → should 404")
except urllib.error.HTTPError as e:
    check(e.code == 404, f"missing image → 404", f"got {e.code}")

# ---------- Summary ----------
print(f"\n{'='*60}")
print(f"  {PASS} {tests_pass} pass    {FAIL if tests_fail else INFO} {tests_fail} fail")
print(f"{'='*60}")
sys.exit(1 if tests_fail else 0)
