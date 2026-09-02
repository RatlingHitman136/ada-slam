const NS = 'http://www.w3.org/2000/svg';
const el = (t, a = {}) => { const n = document.createElementNS(NS, t);
  for (const k in a) n.setAttribute(k, a[k]); return n; };
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const tip = document.getElementById('tip');
const showTip = (html, ev) => { tip.innerHTML = html; tip.style.opacity = 1;
  tip.style.left = Math.min(ev.clientX + 14, innerWidth - 230) + 'px';
  tip.style.top = (ev.clientY - 12) + 'px'; };
const hideTip = () => { tip.style.opacity = 0; };

function frame(svg, m, w, h, xd, yd, xt, yt, xlab, ylab) {
  svg.textContent = '';
  const X = v => m.l + (v - xd[0]) / (xd[1] - xd[0]) * (w - m.l - m.r);
  const Y = v => h - m.b - (v - yd[0]) / (yd[1] - yd[0]) * (h - m.t - m.b);
  const g = el('g'); svg.appendChild(g);
  const ink3 = css('--ink-3'), rule = css('--rule');
  for (const v of yt) {
    g.appendChild(el('line', {x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v),
      stroke: rule, 'stroke-width': 1}));
    const t = el('text', {x: m.l - 10, y: Y(v) + 4, 'text-anchor': 'end', fill: ink3,
      'font-size': 12, 'font-family': css('--mono')}); t.textContent = v; g.appendChild(t);
  }
  for (const v of xt) {
    const t = el('text', {x: X(v), y: h - m.b + 20, 'text-anchor': 'middle', fill: ink3,
      'font-size': 12, 'font-family': css('--mono')}); t.textContent = v; g.appendChild(t);
  }
  g.appendChild(el('line', {x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b,
    stroke: css('--rule-2'), 'stroke-width': 1}));
  const xl = el('text', {x: (m.l + w - m.r) / 2, y: h - 4, 'text-anchor': 'middle', fill: ink3,
    'font-size': 12.5, 'font-family': css('--sans')}); xl.textContent = xlab; g.appendChild(xl);
  const yl = el('text', {x: 14, y: (m.t + h - m.b) / 2, 'text-anchor': 'middle', fill: ink3,
    'font-size': 12.5, 'font-family': css('--sans'),
    transform: `rotate(-90 14 ${(m.t + h - m.b) / 2})`}); yl.textContent = ylab;
  g.appendChild(yl);
  return {g, X, Y};
}

function scatter() {
  const svg = document.getElementById('sc'), w = 900, h = 430;
  const m = {l: 56, r: 26, t: 14, b: 44};
  const xd = [0, 0.45], yd = [0, 30];
  const {g, X, Y} = frame(svg, m, w, h, xd, yd, [0, 0.1, 0.2, 0.3, 0.4],
    [0, 5, 10, 15, 20, 25, 30], 'local Sim(3) scale wobble  (std over a 31-frame window)',
    'ATE (m)');
  g.appendChild(el('line', {x1: X(0), y1: Y(0), x2: X(0.45), y2: Y(Math.min(30, 66 * 0.45)),
    stroke: css('--ink-3'), 'stroke-width': 1.5, 'stroke-dasharray': '5 5', opacity: .7}));
  const cols = [css('--s1'), css('--s2')];
  const pts = [];
  for (const [name, ate, wob, grp] of ARMS) {
    if (wob > xd[1] || ate > yd[1]) continue;
    const c = el('circle', {cx: X(wob), cy: Y(ate), r: 5, fill: cols[grp],
      stroke: css('--panel'), 'stroke-width': 2, opacity: .95});
    c.addEventListener('mouseenter', e => { c.setAttribute('r', 7);
      showTip(`<b>${name}</b><br>ATE ${ate.toFixed(2)} m &middot; wobble ${wob.toFixed(4)}`, e); });
    c.addEventListener('mousemove', e => showTip(tip.innerHTML, e));
    c.addEventListener('mouseleave', () => { c.setAttribute('r', 5); hideTip(); });
    g.appendChild(c); pts.push(c);
  }
  const note = el('text', {x: X(0.36), y: Y(26.5), 'text-anchor': 'start', fill: css('--ink-3'),
    'font-size': 12, 'font-family': css('--sans')});
  note.textContent = '1 arm off scale: wobble 1.92, ATE 45.8';
  g.appendChild(note);
}

function response() {
  const svg = document.getElementById('rc'), w = 900, h = 400;
  const m = {l: 56, r: 96, t: 16, b: 44};
  const T = 1.45, xd = [0, 6], yd = [0, 1.1];
  const {g, X, Y} = frame(svg, m, w, h, xd, yd, [0, 1, 2, 3, 4, 5, 6],
    [0, 0.25, 0.5, 0.75, 1], 'true depth  (× the frame’s median depth)',
    'served depth / true depth');
  g.insertBefore(el('rect', {x: X(0), y: Y(1.1), width: X(1) - X(0), height: Y(0) - Y(1.1),
    fill: css('--accent'), opacity: .07}), g.firstChild);
  const band = el('text', {x: X(0.5), y: Y(1.06), 'text-anchor': 'middle', fill: css('--ink-3'),
    'font-size': 11.5, 'font-family': css('--sans')});
  band.textContent = '82% of the alignment’s leverage'; g.appendChild(band);
  g.appendChild(el('line', {x1: X(T), x2: X(T), y1: Y(0), y2: Y(1.02), stroke: css('--rule-2'),
    'stroke-width': 1, 'stroke-dasharray': '3 4'}));
  const tag = el('text', {x: X(T) + 6, y: Y(1.05), fill: css('--ink-3'), 'font-size': 11.5,
    'font-family': css('--mono')}); tag.textContent = 'tag 1.45'; g.appendChild(tag);
  const series = [
    {n: '@ceil1.45', c: css('--s1'), f: u => Math.min(u, T) / u},
    {n: '@soft1.45', c: css('--s3'), f: u => 1 / Math.sqrt(1 + (u / T) ** 2)},
    {n: '@ped1.45', c: css('--s2'), f: u => 1 / (1 + u / T)}];
  for (const s of series) {
    let d = '';
    for (let i = 0; i <= 300; i++) {
      const u = xd[0] + (xd[1] - xd[0]) * i / 300;
      const y = u < 1e-6 ? 1 : s.f(u);
      d += (i ? 'L' : 'M') + X(u).toFixed(1) + ' ' + Y(y).toFixed(1);
    }
    g.appendChild(el('path', {d, fill: 'none', stroke: s.c, 'stroke-width': 2.2,
      'stroke-linejoin': 'round'}));
    const t = el('text', {x: X(6) + 8, y: Y(s.f(6)) + 4, fill: s.c, 'font-size': 12.5,
      'font-weight': 500, 'font-family': css('--mono')});
    t.textContent = s.n; g.appendChild(t);
  }
  const hit = el('rect', {x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b,
    fill: 'transparent'});
  const cross = el('line', {y1: m.t, y2: h - m.b, stroke: css('--ink-3'), 'stroke-width': 1,
    'stroke-dasharray': '2 3', opacity: 0});
  g.appendChild(cross); g.appendChild(hit);
  hit.addEventListener('mousemove', e => {
    const r = svg.getBoundingClientRect();
    const u = xd[0] + (e.clientX - r.left) / r.width * w > 0
      ? ((e.clientX - r.left) / r.width * w - m.l) / (w - m.l - m.r) * (xd[1] - xd[0]) : 0;
    const uu = Math.max(0.02, Math.min(6, u));
    cross.setAttribute('x1', X(uu)); cross.setAttribute('x2', X(uu));
    cross.setAttribute('opacity', .8);
    showTip(`at <b>${uu.toFixed(2)}×</b> median depth<br>` +
      series.map(s => `${s.n} &rarr; ${(s.f(uu) * 100).toFixed(0)}%`).join('<br>'), e);
  });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('opacity', 0); hideTip(); });
}

scatter(); response();
const mq = matchMedia('(prefers-color-scheme: dark)');
mq.addEventListener('change', () => { scatter(); response(); });
new MutationObserver(() => { scatter(); response(); })
  .observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
