#!/usr/bin/env python3
"""Build an offline, interactive view of saved IID and heading-prior samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def collect(source_dir: Path):
    records = {}
    checks = {}
    for seed in (42, 43):
        path = source_dir / f"seed{seed}_condition_prior_xy.npz"
        with np.load(path) as saved:
            epsilon = saved["epsilon"][..., :2].astype(np.float64)
            source = saved["source"][..., :2].astype(np.float64)
            jitter = saved["angular_jitter"].astype(np.float64)
            active = saved["source_active"]
            assert epsilon.shape == source.shape == (4, 200, 16, 2)
            assert active.all(), "Viewer expects the recorded samples to use the prior."
            assert all(np.isfinite(a).all() for a in (epsilon, source, jitter))
            original_sum, source_sum = epsilon.sum(axis=-2), source.sum(axis=-2)
            original_angle = np.arctan2(original_sum[..., 1], original_sum[..., 0])
            source_angle = np.arctan2(source_sum[..., 1], source_sum[..., 0])
            recovered_heading = wrap(source_angle - jitter)
            heading = np.arctan2(np.sin(recovered_heading).mean(0),
                                 np.cos(recovered_heading).mean(0))
            phi = wrap(source_angle - original_angle)
            c, s = np.cos(phi)[..., None], np.sin(phi)[..., None]
            rotated = np.stack((c * epsilon[..., 0] - s * epsilon[..., 1],
                                s * epsilon[..., 0] + c * epsilon[..., 1]), axis=-1)
            endpoint_error = float(np.max(np.abs(rotated - source)))
            heading_spread = float(np.max(np.abs(wrap(recovered_heading - heading))))
            norm_error = float(np.max(np.abs(np.linalg.norm(epsilon, axis=-1)
                                             - np.linalg.norm(source, axis=-1))))
            assert endpoint_error < 2e-5, endpoint_error
            assert heading_spread < 2e-5, heading_spread
            assert norm_error < 2e-5, norm_error
            assert all(saved[k].shape == (200,) for k in ("episode", "frame", "task"))
            records[str(seed)] = {
                "epsilon": epsilon.tolist(), "source": source.tolist(),
                "jitter": jitter.tolist(), "heading": heading.tolist(),
                "episode": saved["episode"].tolist(), "frame": saved["frame"].tolist(),
                "task": saved["task"].tolist(),
            }
            checks[str(seed)] = {
                "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "shape": list(epsilon.shape), "all_source_active": bool(active.all()),
                "finite": True, "max_endpoint_abs_error": endpoint_error,
                "max_vector_norm_abs_error": norm_error,
                "max_recovered_heading_deviation_rad": heading_spread,
                "lambda_zero_exact": True,
            }
    return records, checks


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IID 与 Heading Prior：噪声对比</title>
<style>
:root{color-scheme:light;--ink:#172b3a;--muted:#526575;--line:#dce5ea;--blue:#2475b5;--orange:#d97020;--green:#28775c}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font-family:Inter,system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;line-height:1.6}main{max-width:1200px;margin:auto;padding:35px 24px 28px}h1{font-size:clamp(25px,3vw,36px);line-height:1.3;letter-spacing:-.5px;margin:0 0 13px;font-weight:720}.intro{max-width:940px;margin:0;color:var(--muted);font-size:16px}.tag{display:inline-block;color:#22664f;background:#e2f0e8;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:650;margin-bottom:12px}.controls{margin-top:25px;background:white;border:1px solid var(--line);border-radius:14px;padding:18px 20px;display:flex;gap:20px;align-items:end;flex-wrap:wrap}label{display:flex;flex-direction:column;font-size:12px;font-weight:650;color:var(--muted);gap:5px}select,input[type=number],button{font:inherit;font-size:15px;color:var(--ink);background:#f9fbfc;border:1px solid #c8d4dc;border-radius:6px;padding:7px 10px;min-height:39px}input[type=number]{width:90px}button{cursor:pointer;min-width:38px}button:hover{background:#eaf0f4}.window-controls{display:flex;gap:6px}.slider-label{flex:1;min-width:230px}.slider-line{display:flex;align-items:center;gap:12px;height:39px}input[type=range]{width:100%;accent-color:var(--orange)}#amount{font-variant-numeric:tabular-nums;min-width:48px;color:var(--ink)}.slider-ends{display:flex;justify-content:space-between;font-size:11px;font-weight:400;margin-top:-5px}.metadata{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:11px 2px 18px;font-variant-numeric:tabular-nums}.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{min-width:0;background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden}.panel-head{padding:17px 20px 0;display:flex;justify-content:space-between;gap:12px;align-items:baseline}.panel h2{font-size:19px;margin:0}.panel small{font-size:12px;color:var(--muted)}.blue{color:var(--blue)}.orange{color:var(--orange)}canvas{display:block;width:100%;aspect-ratio:1/1}.metrics{border-top:1px solid var(--line);display:flex;gap:18px;justify-content:space-between;padding:12px 18px;font-size:12px;color:var(--muted)}.metrics strong{display:block;color:var(--ink);font-size:18px;line-height:1.5;font-weight:650;font-variant-numeric:tabular-nums}.legend{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;padding:15px 8px;color:var(--muted);font-size:12px}.legend span{display:flex;align-items:center;gap:7px}.key{width:23px;border-top:2px solid #5e7182}.key.green{border-top:2px dashed var(--green)}.key.path{border-color:var(--orange)}.callout{background:#eaf1f5;border-radius:10px;padding:16px 20px;font-size:14px;margin-top:5px}.callout p{margin:0}.callout p+p{margin-top:8px}.equation{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--ink);font-size:13px}.foot{font-size:12px;color:var(--muted);margin:16px 2px 0}#status{font-weight:650;color:var(--orange)}@media(max-width:730px){main{padding:22px 14px}.plots{grid-template-columns:1fr}.controls{gap:15px;padding:14px}.slider-label{flex-basis:100%}.panel-head{padding:14px 17px 0}.metrics{gap:10px}.metadata{gap:10px}.legend{justify-content:flex-start;gap:14px}}
</style></head>
<body><main>
<div class="tag">实际实验保存的噪声 · 无需联网</div>
<h1>同一份 IID 噪声，怎样变成方向 prior？</h1>
<p class="intro">左图是原始高斯噪声。右图把整段 16 步的 XY 向量旋转同一个角度，让合成方向朝向「预测 heading + 随机角度扰动」。拖动滑块，观察这个旋转过程。</p>
<section class="controls" aria-label="选择噪声样本">
<label>训练 seed<select id="seed"><option>42</option><option>43</option></select></label>
<label>观测窗口（0–199）<div class="window-controls"><button id="previous" aria-label="上一个观测窗口">‹</button><input id="window" type="number" min="0" max="199" value="0" aria-label="观测窗口"><button id="next" aria-label="下一个观测窗口">›</button></div></label>
<label>同一观测的噪声抽样<select id="draw"><option value="0">第 1 组</option><option value="1">第 2 组</option><option value="2">第 3 组</option><option value="3">第 4 组</option></select></label>
<label class="slider-label">旋转进度<div class="slider-line"><input id="alignment" type="range" min="0" max="100" value="100" aria-label="旋转进度"><output id="amount">100%</output></div><div class="slider-ends"><span>0%：原始 IID</span><span>100%：实际 prior</span></div></label>
</section>
<div class="metadata" id="metadata"></div>
<section class="plots">
<article class="panel"><div class="panel-head"><h2 class="blue">IID Gaussian</h2><small>原始的同一次 ε</small></div><canvas id="iid" role="img" aria-label="IID 噪声的 XY 累加路径"></canvas><div class="metrics"><div>合成方向<strong id="iidAngle"></strong></div><div>距预测 heading<strong id="iidError"></strong></div><div>XY 噪声能量<strong id="iidEnergy"></strong></div></div></article>
<article class="panel"><div class="panel-head"><h2 class="orange">Heading prior</h2><small id="status">实际 prior · 100%</small></div><canvas id="prior" role="img" aria-label="方向先验噪声的 XY 累加路径"></canvas><div class="metrics"><div>合成方向<strong id="priorAngle"></strong></div><div>距预测 heading<strong id="priorError"></strong></div><div>XY 噪声能量<strong id="priorEnergy"></strong></div></div></article>
</section>
<div class="legend"><span><i class="key path"></i>16 步 XY 噪声累加</span><span><i class="key"></i>起点 → 终点：合成向量</span><span><i class="key green"></i>预测 heading（仅表示方向）</span></div>
<div class="callout"><p><b>这些线是噪声的累加示意，不是机器人轨迹，也不是生成的动作。</b>横纵轴采用相同的归一化 XY 尺度；累加仅用于显示整段噪声的合成方向。</p><p id="description"></p><p class="equation" id="formula"></p></div>
<p class="foot">取自 condition + XY prior 实验的 2 个训练 seed × 200 个观测 × 4 次噪声抽样；每份噪声包含 16 步。角度扰动 δ ∼ Von Mises(0, κ=4)。所有展示样本的 prior 均已启用。滑块中间状态用于演示，并非重新运行模型。预测 heading 由保存的 prior 合成方向减去 δ 还原；同一观测的四次抽样一致。</p>
</main>
<script id="samples" type="application/json">__DATA__</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('samples').textContent);
const $ = id => document.getElementById(id);
const wrap = x => Math.atan2(Math.sin(x), Math.cos(x));
const deg = x => `${(wrap(x)*180/Math.PI).toFixed(1)}°`;
const sum = vectors => vectors.reduce((a,v)=>[a[0]+v[0],a[1]+v[1]],[0,0]);
const angle = v => Math.atan2(v[1],v[0]);
const energy = vectors => vectors.reduce((e,v)=>e+v[0]*v[0]+v[1]*v[1],0);
const cumulative = vectors => {const p=[[0,0]];for(const v of vectors){const a=p[p.length-1];p.push([a[0]+v[0],a[1]+v[1]]);}return p;};
const rotate = (vectors,phi) => {const c=Math.cos(phi),s=Math.sin(phi);return vectors.map(v=>[c*v[0]-s*v[1],s*v[0]+c*v[1]]);};
function arrow(ctx,from,to,color,width,dashed=false){const dx=to[0]-from[0],dy=to[1]-from[1],a=Math.atan2(dy,dx);ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[7,5]:[]);ctx.beginPath();ctx.moveTo(...from);ctx.lineTo(...to);ctx.stroke();ctx.setLineDash([]);const n=9;ctx.beginPath();ctx.moveTo(...to);ctx.lineTo(to[0]-n*Math.cos(a-.4),to[1]-n*Math.sin(a-.4));ctx.lineTo(to[0]-n*Math.cos(a+.4),to[1]-n*Math.sin(a+.4));ctx.closePath();ctx.fill();ctx.restore();}
function paint(canvas,points,heading,radius,color){
 const size=canvas.clientWidth,dpr=window.devicePixelRatio||1;canvas.width=Math.round(size*dpr);canvas.height=Math.round(size*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const pad=45,span=size-2*pad,scale=span/(2*radius),center=size/2;
 const xy=p=>[center+p[0]*scale,center-p[1]*scale];
 ctx.font='11px system-ui, sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#7a8b98';
 for(let t=-2;t<=2;t++){const v=t*radius/2,x=center+v*scale,y=center-v*scale;ctx.strokeStyle=t===0?'#c8d4dc':'#e8edf1';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,size-pad);ctx.moveTo(pad,y);ctx.lineTo(size-pad,y);ctx.stroke();ctx.fillText(v.toFixed(1),x,size-pad+17);if(t!==0){ctx.textAlign='right';ctx.fillText(v.toFixed(1),pad-8,y);ctx.textAlign='center';}}
 ctx.fillStyle='#526575';ctx.fillText('X',size-pad+22,center);ctx.fillText('Y',center,pad-20);
 arrow(ctx,xy([0,0]),xy([Math.cos(heading)*radius*.80,Math.sin(heading)*radius*.80]),'#28775c',1.8,true);
 ctx.lineWidth=2.1;ctx.strokeStyle=color;ctx.lineJoin='round';ctx.beginPath();points.forEach((p,i)=>{const v=xy(p);if(i===0)ctx.moveTo(...v);else ctx.lineTo(...v);});ctx.stroke();
 for(let i=1;i<points.length;i++){ctx.beginPath();ctx.arc(...xy(points[i]),2.5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
 arrow(ctx,xy([0,0]),xy(points[points.length-1]),'#5e7182',1.4);
 ctx.beginPath();ctx.arc(...xy([0,0]),4.5,0,Math.PI*2);ctx.fillStyle='#172b3a';ctx.fill();
 const finish=xy(points[points.length-1]);ctx.beginPath();ctx.arc(...finish,5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();
 ctx.fillStyle='#526575';ctx.textAlign='left';ctx.fillText('0',center+8,center+12);ctx.fillText('16',finish[0]+8,finish[1]-9);
}
function render(){
 const seed=$('seed').value,win=Math.max(0,Math.min(199,Math.round(Number($('window').value)||0))),draw=Number($('draw').value),lambda=Number($('alignment').value)/100;$('window').value=win;
 const d=DATA[seed],eps=d.epsilon[draw][win],source=d.source[draw][win],heading=d.heading[win],jitter=d.jitter[draw][win],iidAngle=angle(sum(eps)),targetAngle=angle(sum(source)),phi=wrap(targetAngle-iidAngle);
 const current=lambda===0?eps:lambda===1?source:rotate(eps,phi*lambda);const currentAngle=angle(sum(current));const left=cumulative(eps),right=cumulative(current);const maxRadius=Math.max(...left.map(p=>Math.hypot(...p)),...cumulative(source).map(p=>Math.hypot(...p)));const radius=Math.max(1,Math.ceil(maxRadius*1.12));
 paint($('iid'),left,heading,radius,'#2475b5');paint($('prior'),right,heading,radius,'#d97020');
 $('amount').value=`${Math.round(lambda*100)}%`;$('status').textContent=lambda===1?'实际 prior · 100%':lambda===0?'原始 IID · 0%':`旋转中 · ${Math.round(lambda*100)}%`;
 $('metadata').textContent=`Seed ${seed}　·　Window ${win}　·　Episode ${d.episode[win]}　·　Frame ${d.frame[win]}　·　Task ${d.task[win]}　·　预测 heading ${deg(heading)}　·　扰动 δ ${deg(jitter)}`;
 $('iidAngle').textContent=deg(iidAngle);$('priorAngle').textContent=deg(currentAngle);$('iidError').textContent=deg(Math.abs(wrap(iidAngle-heading)));$('priorError').textContent=deg(Math.abs(wrap(currentAngle-heading)));$('iidEnergy').textContent=energy(eps).toFixed(2);$('priorEnergy').textContent=energy(current).toFixed(2);
 $('description').textContent=`完整对齐会将整段旋转 ${deg(phi)}，使合成方向变成 ${deg(targetAngle)}。每一步的 XY 向量长度保持不变，整段噪声的形状也保持不变；不是把每一步都改成同一方向。`;
 $('formula').textContent=`θ目标 = ${deg(heading)} + (${deg(jitter)}) = ${deg(targetAngle)}　|　φ = θ目标 − θIID = ${deg(phi)} （角度按 360° 取等价值）`;
}
for(const id of ['seed','draw','window','alignment'])$(id).addEventListener('input',render);
$('previous').addEventListener('click',()=>{$('window').value=(Number($('window').value)+199)%200;render();});
$('next').addEventListener('click',()=>{$('window').value=(Number($('window').value)+1)%200;render();});
window.addEventListener('resize',render);render();
</script></body></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=ROOT / "output/mini_heading_prior/sampling_robustness_20260909")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output/mini_heading_prior/noise_visualization_20260909")
    args = parser.parse_args()
    records, checks = collect(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Full saved floating-point values preserve the slider's exact endpoint samples.
    html = HTML.replace("__DATA__", json.dumps(records, separators=(",", ":"), allow_nan=False))
    output = args.output_dir / "interactive.html"
    output.write_text(html, encoding="utf-8")
    validation = {"seeds": checks, "offline": True, "external_resources": [],
                  "coordinates": "normalized XY; min-max scale=1.0667 shared, offset=0",
                  "interpolation": "shortest whole-chunk XY rotation; endpoints use saved arrays",
                  "source_file": str(Path(__file__).relative_to(ROOT))}
    (args.output_dir / "interactive_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
