// static/ui.js
(() => {
  const $ = (id) => document.getElementById(id);
  const nlEl = $('nl'), nlOut = $('nlout'), runOut = $('runout'), narOut = $('narout');
  const s1 = $('s1'), s2 = $('s2'), s3 = $('s3');
  const badge1 = $('badgeStep1'), badge2 = $('badgeStep2'), badge3 = $('badgeStep3');
  const modelName = $('modelName'), predLen = $('predLen'), riskBadge = $('riskBadge');
  const API = (window.API_BASE || '');

  let parsed = null, spec = null, lastRun = null, chart = null;

  function showJSON(el, obj) { try { el.textContent = JSON.stringify(obj, null, 2); } catch(e){ el.textContent = String(obj);} }
  function setBadge(el, text, cls) { el.className = 'badge ' + cls; el.textContent = text; }
  function setLoading(span){ span.innerHTML = '<span class="spinner"></span> Processing...'; }
  function clear(el){ el.textContent = ''; }

  function copyFrom(targetId){
    const el = document.getElementById(targetId);
    const txt = el?.textContent || '';
    navigator.clipboard.writeText(txt).then(()=> {
      const btns = document.querySelectorAll(`[data-target="${targetId}"]`);
      btns.forEach(b => { const old=b.textContent; b.textContent='Copied!'; setTimeout(()=>b.textContent='Copy',900); });
    });
  }
  document.querySelectorAll('.copy-btn').forEach(b => b.addEventListener('click', () => copyFrom(b.dataset.target)));

  async function postJSON(path, body) {
    const r = await fetch(API + path, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const j = await r.json().catch(()=> ({}));
    return { ok: r.ok, json: j };
  }

  $('btnNL').onclick = async () => {
    setBadge(badge1,'Running','badge-soft'); setLoading(s1); clear(nlOut);
    try {
      const r = await postJSON('/api/v1/prediction/request-processing', { query: nlEl.value });
      if (!r.ok) { setBadge(badge1,'Failed','badge-danger'); s1.textContent='실패'; showJSON(nlOut, r.json); return; }
      parsed = r.json; spec = parsed.data;
      setBadge(badge1,'Done','badge-ok'); s1.textContent='완료'; showJSON(nlOut, parsed);
    } catch (e) {
      setBadge(badge1,'Error','badge-danger'); s1.textContent='에러'; nlOut.textContent=String(e);
    }
  };

  function drawChart(pred){
    try {
      const ctx = document.getElementById('predChart');
      if (chart) chart.destroy();
      chart = new Chart(ctx, {
        type: 'line',
        data: { labels: pred.map((_,i)=>`t+${i+1}`), datasets: [{ label: 'Prediction', data: pred }] },
        options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ display:true } } }
      });
    } catch {}
  }

  $('btnRun').onclick = async () => {
    if (!spec) { alert('먼저 NL→JSON 하세요'); return; }
    setBadge(badge2,'Running','badge-soft'); setLoading(s2); clear(runOut);
    try {
      const payload = {
        taskId: spec.taskId, timeRange: spec.timeRange,
        sensor_name: spec.sensor_name, target_cols: spec.target_cols,
        fromAgent: 'ui', objective: 'prediction'
      };
      const r = await postJSON('/api/v1/prediction/run-direct', payload);
      lastRun = r.json; showJSON(runOut, r.json);
      s2.textContent = r.ok ? '완료' : '실패';

      const d = r.json?.data || {};
      modelName.textContent = d.modelSelected ?? '-';
      predLen.textContent   = d.pred_len ?? '-';
      const level = d?.risk?.riskLevel || 'unknown';
      let cls = 'badge-soft';
      if (['normal','low'].includes(level)) cls='badge-ok';
      else if (level === 'medium') cls='badge-warn';
      else if (['high','critical'].includes(level)) cls='badge-danger';
      setBadge(riskBadge, level, cls);

      if (Array.isArray(d?.prediction) && d.prediction.length) drawChart(d.prediction);
      setBadge(badge2, r.ok ? 'Done' : 'Failed', r.ok ? 'badge-ok' : 'badge-danger');
    } catch (e) {
      setBadge(badge2,'Error','badge-danger'); s2.textContent='에러'; runOut.textContent=String(e);
    }
  };

  // narrate는 서버가 내부에서 run-direct를 재실행 (항상 run 이후)
  $('btnNarr').onclick = async () => {
    if (!spec) { alert('먼저 NL→JSON과 Run을 실행하세요'); return; }
    setBadge(badge3,'Running','badge-soft'); setLoading(s3); clear(narOut);
    try {
      const runReq = {
        taskId: spec.taskId, timeRange: spec.timeRange,
        sensor_name: spec.sensor_name, target_cols: spec.target_cols,
        fromAgent: 'ui', objective: 'prediction'
      };
      const r = await postJSON('/api/v1/prediction/narrate_from_run', runReq);
      s3.textContent = r.ok ? '완료' : '실패';
      narOut.textContent = r.json?.data?.narration || JSON.stringify(r.json, null, 2);
      setBadge(badge3, r.ok ? 'Done' : 'Failed', r.ok ? 'badge-ok' : 'badge-danger');
    } catch (e) {
      setBadge(badge3,'Error','badge-danger'); s3.textContent='에러'; narOut.textContent=String(e);
    }
  };
})();
