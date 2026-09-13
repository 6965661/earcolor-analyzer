import json
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

app = FastAPI(title='EarColor Link Reader')

NOTE_TO_PC = {'C':0,'C#':1,'DB':1,'D':2,'D#':3,'EB':3,'E':4,'F':5,'F#':6,'GB':6,'G':7,'G#':8,'AB':8,'A':9,'A#':10,'BB':10,'B':11}
PC_TO_NOTE_SHARP = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
DEGREE_COLORS = {1:'#ff565b',2:'#ff951e',3:'#ffd916',4:'#39c66f',5:'#4f7f9f',6:'#2d468e',7:'#6d4a73'}
MAJOR_DEGREES = {0:1,2:2,4:3,5:4,7:5,9:6,11:7}
MINOR_DEGREES = {0:1,2:2,3:3,5:4,7:5,8:6,10:7}

CHORD_RE = re.compile(r'\b([A-G](?:#|b)?(?:m|maj|min|dim|aug|sus)?(?:2|4|5|6|7|9|11|13)?(?:add\d+)?(?:/[A-G](?:#|b)?)?)\b')
KEY_RE = re.compile(r'\b([A-G](?:#|b)?)\s*(major|minor|maj|min)\b', re.I)
BPM_RE = re.compile(r'\b(?:bpm|tempo)\D{0,12}(\d{2,3}(?:\.\d+)?)', re.I)

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EarColor Link</title><style>
:root{color-scheme:dark}body{margin:0;background:#0b0d11;color:#f4f7fb;font-family:Inter,system-ui,Arial,sans-serif}.wrap{max-width:980px;margin:0 auto;padding:34px 18px 70px}.brand{font-size:13px;letter-spacing:.22em;color:#8f9aab;text-transform:uppercase}.hero{font-size:42px;line-height:1.03;margin:10px 0 8px}.sub{color:#aeb7c4;max-width:720px}.row{display:flex;gap:10px;margin:26px 0;flex-wrap:wrap}input,button,select{border:1px solid #252a33;background:#12151b;color:white;border-radius:12px;padding:14px 16px;font-size:16px}input{flex:1;min-width:260px}button{cursor:pointer;background:#1b67ff;border-color:#1b67ff;font-weight:700}.card{background:#12151b;border:1px solid #242a34;border-radius:18px;padding:18px;margin-top:18px}.meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.meta div{background:#0d1015;border-radius:12px;padding:14px}.label{font-size:11px;letter-spacing:.16em;color:#7f8998;text-transform:uppercase}.value{font-size:21px;margin-top:5px}.trans{display:flex;align-items:center;gap:10px;margin-top:16px}.trans button{width:46px;height:42px;padding:0}.chords{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}.chip{border:1px solid #2b313c;border-radius:12px;padding:10px 12px;min-width:72px}.ch{font-size:20px;font-weight:800}.deg{font-size:12px;color:#c7d0db;margin-top:4px}.dot{width:11px;height:11px;border-radius:50%;display:inline-block;margin-right:6px}.error{color:#ff8b8b}.muted{color:#8f9aab}.raw{white-space:pre-wrap;font:12px ui-monospace,monospace;color:#929baa;max-height:250px;overflow:auto} @media(max-width:700px){.hero{font-size:34px}.meta{grid-template-columns:1fr}.row{display:block}.row input,.row button{width:100%;box-sizing:border-box;margin-bottom:10px}}</style></head><body><div class="wrap"><div class="brand">EarColor / Link Mode</div><h1 class="hero">Usa los datos de ChordU,<br>pero míralos por función y color.</h1><p class="sub">Pega un enlace público de ChordU. La página intenta leer tonalidad, BPM y acordes disponibles, y los convierte a posición Movable Do + color. Puedes transponer sin cambiar la función armónica.</p><div class="row"><input id="url" value="https://chordu.com/chords-tabs-dirt-id_IxhDbQWP0Cs"><button onclick="loadSong()">Leer canción</button></div><div id="status" class="muted"></div><div id="out"></div></div><script>
let data=null, shift=0;
const names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function pc(note){let n=note.replace('♭','b').replace('♯','#');let m={'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11};return m[n]}
function rootOf(c){let m=(c||'').match(/^([A-G](?:#|b)?)/);return m?m[1]:null}
function qualityTail(c){let m=(c||'').match(/^[A-G](?:#|b)?(.*)$/);return m?m[1]:''}
function transposeChord(c,s){let r=rootOf(c);if(!r)return c;let p=pc(r);if(p==null)return c;return names[(p+s+120)%12]+qualityTail(c)}
function render(){if(!data)return;let keyRoot=data.key_root;let keyMode=data.key_mode||'major';let tkey=keyRoot?names[(pc(keyRoot)+shift+120)%12]:null;let scale=keyMode==='minor'?{0:1,2:2,3:3,5:4,7:5,8:6,10:7}:{0:1,2:2,4:3,5:4,7:5,9:6,11:7};let colors={1:'#ff565b',2:'#ff951e',3:'#ffd916',4:'#39c66f',5:'#4f7f9f',6:'#2d468e',7:'#6d4a73'};let chips=data.chords.map(c=>{let tc=transposeChord(c,shift),r=rootOf(c),d='—',col='#6b7280';if(r&&keyRoot){let rel=(pc(r)-pc(keyRoot)+12)%12;if(scale[rel]){d=scale[rel];col=colors[d]}}return `<div class="chip"><div class="ch">${tc}</div><div class="deg"><span class="dot" style="background:${col}"></span>Posición ${d}</div></div>`}).join('');document.getElementById('out').innerHTML=`<div class="card"><div class="meta"><div><div class="label">Canción</div><div class="value">${data.title||'—'}</div></div><div><div class="label">Tonalidad</div><div class="value">${tkey? tkey+' '+keyMode:'No detectada'}</div></div><div><div class="label">BPM</div><div class="value">${data.bpm||'—'}</div></div></div><div class="trans"><button onclick="shift--;render()">−</button><div><div class="label">Transposición</div><div class="value">${shift>0?'+':''}${shift} semitonos</div></div><button onclick="shift++;render()">+</button></div></div><div class="card"><div class="label">Acordes encontrados</div><div class="chords">${chips||'<span class="muted">No se encontraron acordes estructurados en el HTML público.</span>'}</div></div><div class="card"><div class="label">Diagnóstico de extracción</div><div class="raw">${data.notes.join('\n')}</div></div>`}
async function loadSong(){let u=document.getElementById('url').value.trim();document.getElementById('status').textContent='Leyendo página pública…';document.getElementById('out').innerHTML='';try{let r=await fetch('/api/read?url='+encodeURIComponent(u));let j=await r.json();if(!r.ok)throw new Error(j.detail||'Error');data=j;shift=0;document.getElementById('status').textContent=`Encontrados ${j.chords.length} acordes únicos.`;render()}catch(e){document.getElementById('status').innerHTML='<span class="error">'+e.message+'</span>'}}
</script></body></html>'''


def flatten_text(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            flatten_text(v, out)
    elif isinstance(obj, list):
        for x in obj:
            flatten_text(x, out)
    elif isinstance(obj, (str, int, float)):
        out.append(str(obj))


def normalize_chord(token: str):
    token = token.strip().replace('♭', 'b').replace('♯', '#')
    if len(token) > 12:
        return None
    return token


@app.get('/', response_class=HTMLResponse)
def home():
    return HTML


@app.get('/health')
def health():
    return {'ok': True, 'service': 'earcolor-link'}


@app.get('/api/read')
async def read_page(url: str = Query(...)):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or parsed.netloc.lower() not in ('chordu.com', 'www.chordu.com'):
        raise HTTPException(400, 'Por ahora solo acepto enlaces de chordu.com')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/149 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20, headers=headers) as client:
            resp = await client.get(url)
        if resp.status_code >= 400:
            raise HTTPException(502, f'ChordU respondió HTTP {resp.status_code}')
    except httpx.HTTPError as e:
        raise HTTPException(502, f'No pude leer ChordU: {e}')

    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og = soup.find('meta', attrs={'property':'og:title'})
    if og and og.get('content'):
        title = og['content'].strip()

    notes = [f'HTTP {resp.status_code}', f'HTML recibido: {len(html):,} caracteres']
    text_pool = []
    visible = soup.get_text(' ', strip=True)
    text_pool.append(visible)

    json_objects = []
    for script in soup.find_all('script'):
        txt = script.string or script.get_text() or ''
        stype = (script.get('type') or '').lower()
        sid = script.get('id') or ''
        if stype == 'application/ld+json' or sid == '__NEXT_DATA__' or txt.lstrip().startswith(('{','[')):
            try:
                obj = json.loads(txt)
                json_objects.append(obj)
            except Exception:
                pass
        if any(k in txt.lower() for k in ('chord','tempo','bpm','key')):
            text_pool.append(txt)

    if json_objects:
        flat=[]
        for obj in json_objects:
            flatten_text(obj, flat)
        text_pool.append(' '.join(flat))
        notes.append(f'JSON estructurado detectado: {len(json_objects)} bloque(s)')
    else:
        notes.append('No encontré JSON estructurado directamente; usando extracción por texto/scripts.')

    corpus='\n'.join(text_pool)

    key_root = None
    key_mode = None
    km = KEY_RE.search(corpus)
    if km:
        key_root = km.group(1).replace('♭','b').replace('♯','#')
        key_mode = 'minor' if km.group(2).lower() in ('minor','min') else 'major'
        notes.append(f'Tonalidad detectada: {key_root} {key_mode}')
    else:
        # common JSON fields like "key":"C#m"
        m = re.search(r'["\'](?:key|tonic|scale)["\']\s*[:=]\s*["\']([A-G](?:#|b)?)(m|minor|major)?', corpus, re.I)
        if m:
            key_root=m.group(1); key_mode='minor' if (m.group(2) or '').lower() in ('m','minor') else 'major'
            notes.append(f'Tonalidad detectada por campo: {key_root} {key_mode}')

    bpm = None
    bm = BPM_RE.search(corpus)
    if bm:
        bpm = float(bm.group(1))
        if bpm.is_integer(): bpm = int(bpm)
        notes.append(f'BPM detectado: {bpm}')

    # Prefer chord-like arrays/fields, then fallback to all chord tokens in chord-related script text.
    chord_candidates=[]
    for pat in [
        r'["\'](?:chord|chords|progression|chordSequence)["\']\s*[:=]\s*\[([^\]]{1,20000})\]',
        r'["\'](?:chord|name|label)["\']\s*[:=]\s*["\']([A-G](?:#|b)?[^"\']{0,10})["\']',
    ]:
        for mm in re.finditer(pat, corpus, re.I|re.S):
            chunk=mm.group(1)
            chord_candidates += [normalize_chord(x.group(1)) for x in CHORD_RE.finditer(chunk)]

    if len([c for c in chord_candidates if c]) < 2:
        chord_related='\n'.join(t for t in text_pool if 'chord' in t.lower())
        chord_candidates += [normalize_chord(x.group(1)) for x in CHORD_RE.finditer(chord_related)]

    # Filter obvious English false positives and dedupe while preserving order.
    blacklist={'A','Am'} if False else set()
    chords=[]
    seen=set()
    for c in chord_candidates:
        if not c or c in blacklist: continue
        # reject bare article-like single A unless surrounded by chord context is unknown; keep because A is valid chord
        if c not in seen:
            seen.add(c); chords.append(c)
        if len(chords)>=80: break

    notes.append(f'Acordes únicos extraídos: {len(chords)}')
    if not chords:
        notes.append('La página puede cargar la progresión dinámicamente después del HTML inicial. En ese caso habrá que leer la llamada interna que usa su reproductor.')

    return {'source_url': str(resp.url), 'title': title, 'key_root': key_root, 'key_mode': key_mode, 'bpm': bpm, 'chords': chords, 'notes': notes}
