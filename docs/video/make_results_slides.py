# Results slides for the video, in the architecture diagram's style. 1920x1080.
CSS = """
* { box-sizing:border-box; margin:0; }
body { width:1920px; height:1080px; background:#ffffff; font-family:Lato, sans-serif; color:#1f1e1d;
       padding:84px 96px; display:flex; flex-direction:column; }
h1 { font-size:46px; font-weight:700; letter-spacing:-0.5px; }
.sub { font-size:24px; color:#6f6c66; margin-top:12px; }
.grid { flex:1; display:grid; gap:32px; margin-top:52px; }
.grid.four { grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; }
.grid.one { grid-template-columns:1fr; }
.card { border:2px solid #c9c6bf; border-radius:22px; padding:36px 44px; display:flex; flex-direction:column; }
.card.accent { border:3px solid #2f6fd0; background:#eef4fd; }
.label { font-size:20px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#6f6c66; }
.big { font-size:96px; font-weight:900; letter-spacing:-2px; line-height:1; margin-top:18px; }
.big .unit { font-size:44px; font-weight:700; letter-spacing:0; color:#6f6c66; margin-left:6px; }
.blue { color:#2f6fd0; }
.note { font-size:22px; color:#6f6c66; margin-top:auto; padding-top:16px; }
.bars { margin-top:22px; display:flex; flex-direction:column; gap:14px; }
.bar { display:grid; grid-template-columns:240px 1fr 110px; align-items:center; gap:16px; font-size:24px; }
.bar .track { height:30px; }
.bar .fill { height:100%; border-radius:8px; background:#c9c6bf; }
.bar.hi .fill { background:#2f6fd0; }
.bar .v { font-weight:700; text-align:right; }
.bar.hi .v { color:#2f6fd0; }
.row2 { display:flex; gap:56px; align-items:flex-end; margin-top:18px; }
.small { font-size:56px; font-weight:900; line-height:1; }
.cap { font-size:22px; color:#6f6c66; margin-top:10px; }
.one .cap { font-size:30px; }
.small .unit { font-size:24px; color:#6f6c66; font-weight:700; margin-left:4px; }
.one .card { padding:60px 80px; justify-content:center; }
.one .big { font-size:150px; }
.one .big .unit { font-size:64px; }
.one .bar { grid-template-columns:330px 1fr 150px; font-size:34px; }
.one .bar .track { height:46px; }
.one .label { font-size:28px; }
.one .note { font-size:30px; margin-top:40px; }
.one .small { font-size:64px; } .one .small .unit { font-size:34px; }
"""

def bars(rows, vmax):
    return '<div class="bars">' + "".join(
        f'<div class="bar{" hi" if hi else ""}"><span>{name}</span><div class="track"><div class="fill" '
        f'style="width:{v / vmax * 100:.1f}%"></div></div><span class="v">{v:g} s</span></div>'
        for name, v, hi in rows) + "</div>"

CARDS = {
 "latency": f'''<div class="card accent"><div class="label">Camera to result, from the real car</div>
   <div class="big blue">1.3<span class="unit">s</span></div>
   {bars([("first live run", 3.2, False), ("now", 1.3, True)], 3.2)}
   <div class="note">Pi 3B in the car → S3 + SQS → OpenCV 5 on Graviton → live page</div></div>''',
 "inference": f'''<div class="card"><div class="label">YOLOPv2 inference per frame</div>
   {bars([("PyTorch, 4 cores", 0.89, False), ("OpenCV 5, 4 cores", 0.74, True),
          ("OpenCV 5, 8 cores", 0.42, True), ("OpenCV 5, 16 cores", 0.22, True)], 0.89)}
   <div class="note">cv.dnn on COOL, Graviton4 c8g instances</div></div>''',
 "lanes": '''<div class="card"><div class="label">Lanes, against hand-labelled frames</div>
   <div class="row2"><div><div class="big blue">98<span class="unit">%</span></div><div class="cap">right when it reports a lane</div></div>
   <div><div class="small">73<span class="unit">%</span></div><div class="cap">of visible lanes found</div></div></div>
   <div class="note">95 frames, labelled by hand without seeing the pipeline's answer</div></div>''',
 "signs": '''<div class="card"><div class="label">Stop signs</div>
   <div class="row2"><div class="big blue">0</div><div class="small">false alarms<span class="unit"> in 8,001 frames</span></div></div>
   <div class="note">Speed limits: 99% read at 85 px and wider, 3 misreads in 1,964 signs</div></div>''',
}

def page(title, sub, cards, layout):
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>'
            f'<h1>{title}</h1><div class="sub">{sub}</div><div class="grid {layout}">{"".join(cards)}</div></body></html>')

SUB = "Initiative Vision · measured October 2026"
open("docs/video/results.html", "w").write(page("Results", SUB, list(CARDS.values()), "four"))
titles = {"latency": "Faster", "inference": "OpenCV 5 on Graviton", "lanes": "Lanes", "signs": "Signs"}
for k, card in CARDS.items():
    open(f"docs/video/results-{k}.html", "w").write(page(titles[k], SUB, [card], "one"))

# Render (from the repo root):
#   python docs/video/make_results_slides.py
#   for f in results results-latency results-inference results-lanes results-signs; do
#     google-chrome --headless=new --hide-scrollbars --window-size=1920,1080 \
#       --screenshot=docs/video/$f.png file://$PWD/docs/video/$f.html; done
