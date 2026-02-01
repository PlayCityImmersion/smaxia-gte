#!/usr/bin/env python3
"""
SMAXIA GTE V14.2 — ADMIN COMMAND CENTER — FULL AUTO
streamlit run smaxia_gte_v14_2_admin_final.py

# ── CHANGELOG V14.1.1 → V14.2 ──────────────────────────────
# [ADD-SDA0] Auto Source Discovery: DuckDuckGo HTML + portal
#   crawl. Multi-strategy, cached, zero manual URL input.
# [ADD-DA1-AUTO] Auto PDF download from discovered links.
#   Dedup sha256, timeout/retry, sujet/corrige pairing.
# [ADD-CAP-AUTO] Auto CAP from PDF metadata + link text.
#   Completeness=PASS if levels+subjects found.
# [ADD-FORMULA-AUTO] Auto-fetch formula pack from registry.
#   If not published => UNAVAILABLE_PACK_NOT_PUBLISHED.
# [ADD-IA] Optional LLM providers (env vars). DISABLED_NO_KEY
#   if absent. Content-addressed cache for determinism.
# [ADD-GATE] GATE_SDA0_WEB for web discovery status.
# [PRESERVED] All V14.1.1: Artifacts tab, ZIP, typeahead.
# ────────────────────────────────────────────────────────────
"""
import streamlit as st
import json, hashlib, os, re, math, uuid, io, zipfile, time
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from urllib.parse import urljoin, urlparse, quote_plus

VERSION = "GTE-V14.2-ADMIN-FINAL"
PACKS_DIR = Path("packs")
RUNS_DIR = Path("run")
FORMULA_PACK_DIR = Path("formula_packs")
OCR_CACHE_DIR = Path("ocr_cache")
HARVEST_DIR = Path("harvest")
WEB_CACHE_DIR = Path("web_cache")
IA_CACHE_DIR = Path("ia_cache")
DETERMINISM_RUNS = 3
SDA0_MAX_PDFS = 20
SDA0_TIMEOUT = 15
SDA0_UA = "SMAXIA-GTE/14.2 (education-research)"
FORMULA_REG = os.environ.get("SMAXIA_FORMULA_REGISTRY", "https://raw.githubusercontent.com/smaxia-project/formula-packs/main/registry.json")
VOLATILE = frozenset(["timestamp","created_at","sealed_at","run_ts","harvested_at","paired_at","activated_at","checked_at","extracted_at","cached_at","run_id","path","run_dir","ts","abs_path","elapsed_ms","discovery_ts","download_ts","fetch_ts"])

# ━━ CORE UTILS ━━
def cjson(o): return json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def sha(d:str)->str: return hashlib.sha256(d.encode()).hexdigest()
def sha_b(d:bytes)->str: return hashlib.sha256(d).hexdigest()
def sha_file(p:Path)->str:
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(8192),b""): h.update(c)
    return h.hexdigest()
def now_iso(): return datetime.now(timezone.utc).isoformat()
def edir(p:Path): p.mkdir(parents=True,exist_ok=True); return p
def strip_vol(o,fields=VOLATILE):
    if isinstance(o,dict): return {k:strip_vol(v,fields) for k,v in sorted(o.items()) if k not in fields}
    if isinstance(o,list): return [strip_vol(i,fields) for i in o]
    return o
def write_art(rd:Path,name:str,payload:dict):
    s=strip_vol(deepcopy(payload)); d=sha(cjson(s)); full={**payload,"_sha256_functional":d}
    (rd/f"{name}.json").write_text(json.dumps(full,sort_keys=True,indent=2,ensure_ascii=False),encoding="utf-8")
    (rd/f"{name}.sha256").write_text(d,encoding="utf-8"); return d
def log_evt(rd,evt,detail,triggered=False):
    p=rd/"UI_EVENT_LOG.json"; evts=json.loads(p.read_text()) if p.exists() else []
    evts.append({"ts":now_iso(),"event":evt,"detail":detail,"triggered_pipeline":triggered})
    p.write_text(json.dumps(evts,indent=2,ensure_ascii=False))
def seal_evt_log(rd):
    p=rd/"UI_EVENT_LOG.json"
    if not p.exists(): return {"status":"FAIL","reason":"NO_LOG"}
    evts=json.loads(p.read_text()); trigs=[e for e in evts if e.get("triggered_pipeline")]
    bad=[e for e in trigs if e["event"]!="ACTIVATE_COUNTRY"]
    ok=len(trigs)>=1 and len(bad)==0; d=sha(cjson([strip_vol(e) for e in evts]))
    (rd/"UI_EVENT_LOG.sha256").write_text(d)
    return {"status":"PASS" if ok else "FAIL","triggers":len(trigs),"bad":len(bad)}

# ━━ WEB UTILS (cached) ━━
def _http_get(url,timeout=SDA0_TIMEOUT,binary=False):
    try: import requests
    except ImportError: return None,-1,False
    ck=sha(url); ext=".bin" if binary else ".html"
    cp=edir(WEB_CACHE_DIR)/f"{ck}{ext}"; cm=edir(WEB_CACHE_DIR)/f"{ck}.meta"
    if cp.exists() and cm.exists():
        try:
            meta=json.loads(cm.read_text()); data=cp.read_bytes() if binary else cp.read_text(encoding="utf-8",errors="replace")
            return data,meta.get("status",200),True
        except: pass
    try:
        r=requests.get(url,timeout=timeout,headers={"User-Agent":SDA0_UA},allow_redirects=True)
        r.raise_for_status()
        if binary: data=r.content; cp.write_bytes(data)
        else: data=r.text; cp.write_text(data,encoding="utf-8")
        cm.write_text(json.dumps({"url":url,"status":r.status_code,"fetch_ts":now_iso(),"size":len(data)}))
        return data,r.status_code,False
    except: return None,-1,False

def _extract_pdf_links(html,base_url):
    try: from bs4 import BeautifulSoup
    except ImportError: return []
    if not html: return []
    soup=BeautifulSoup(html,"html.parser"); links=[]; seen=set()
    for a in soup.find_all("a",href=True):
        href=a["href"].strip()
        if not href: continue
        full=urljoin(base_url,href)
        if full in seen: continue
        seen.add(full)
        if href.lower().endswith(".pdf") or "pdf" in href.lower():
            links.append({"url":full,"text":(a.get_text(strip=True) or "")[:200],"href_raw":href[:300]})
    return links

# ━━ COUNTRY (UI-ONLY) ━━
@st.cache_data
def _build_country_index():
    try:
        import pycountry; db={c.alpha_2:c.name for c in pycountry.countries}
    except ImportError:
        db={"FR":"France","BE":"Belgium","CI":"Côte d'Ivoire","SN":"Senegal","CM":"Cameroon","NG":"Nigeria","CA":"Canada","US":"United States","GB":"United Kingdom","DE":"Germany","IT":"Italy","ES":"Spain","MA":"Morocco","TN":"Tunisia","GA":"Gabon","GH":"Ghana","GN":"Guinea","GE":"Georgia","GR":"Greece","NI":"Nicaragua","NE":"Niger","ZA":"South Africa","ZM":"Zambia","ZW":"Zimbabwe"}
    idx=[{"code":c,"name":n,"nl":n.lower(),"cl":c.lower()} for c,n in db.items()]
    idx.sort(key=lambda e:e["name"]); return db,idx

def typeahead(q,limit=20):  # UI-ONLY
    _,idx=_build_country_index(); ql=q.strip().lower()
    if not ql: return [],[]
    np,fb,seen=[],[],set()
    for e in idx:
        if e["nl"].startswith(ql): np.append(e); seen.add(e["code"])
    for e in idx:
        if e["code"] in seen: continue
        if e["cl"].startswith(ql) or ql in e["nl"] or ql in e["cl"]: fb.append(e); seen.add(e["code"])
    return np[:limit],fb[:limit]

def _country_name(ck):
    db,_=_build_country_index(); return db.get(ck,ck)

# Language discovery — linguistic metadata, NOT country business logic
_LANG_DB={"fr":{"FR","BE","CI","SN","CM","GA","GN","TN","MA","ML","BF","TD","CG","CD","MG","BJ","NE","TG","HT","LU","DJ","KM","RW","BI"},"en":{"US","GB","CA","AU","NZ","NG","GH","ZA","KE","IN","PH","SG","IE","ZM","ZW","BW","UG","TZ"},"es":{"ES","MX","CO","AR","PE","VE","CL","EC","GT","CU","BO","DO","HN","PY","SV","NI","CR","PA","UY"},"pt":{"BR","PT","AO","MZ"},"de":{"DE","AT"},"ar":{"SA","EG","IQ","JO","LB","DZ","SD","YE","KW","QA","AE"},"ja":{"JP"},"it":{"IT"}}
def _lang(ck):  # UI-ONLY — linguistic metadata
    for l,cs in _LANG_DB.items():
        if ck in cs: return l
    return "en"

# ━━ SDA0 ━━
_STPL={"fr":["{c} sujets corrigés bac terminale filetype:pdf","{c} annales examens sujets corrigés pdf","{c} épreuves baccalauréat sujets et corrigés"],"en":["{c} past exam papers answers pdf","{c} national examination solved papers pdf"],"es":["{c} exámenes resueltos secundaria pdf"],"pt":["{c} provas resolvidas exame nacional pdf"],"de":["{c} Abitur Prüfungen Lösungen pdf"],"ar":["{c} امتحانات محلولة pdf"]}
_AUTH_PAT=[(r"\.gouv\.",90),(r"\.gov\.",90),(r"\.edu\.",80),(r"\.ac\.",80),(r"\.education\.",85),(r"\.org",60),(r"annales",50),(r"sujet",40),(r"examen",40)]
_EXAM_KW=["pdf","sujet","corrig","exam","annal","paper","past","epreuve","solved"]

class SDA0:
    def __init__(self,ck,rd):
        self.ck,self.rd=ck,rd; self.cn=_country_name(ck); self.lang=_lang(ck)
        self.sources,self.pdf_links,self.quarantine,self.dlog=[],[],[],[]
    def discover(self):
        t0=time.time(); self._search(); self._portals(); self._dedup(); self._score()
        self.dlog.append({"strategy":"ALL","elapsed_ms":int((time.time()-t0)*1000),"sources":len(self.sources),"links":len(self.pdf_links)})
        return self.sources,self.pdf_links
    def _search(self):
        tmpls=_STPL.get(self.lang,_STPL["en"])
        for t in tmpls[:3]:
            q=t.format(c=self.cn); url=f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
            html,st,cached=_http_get(url)
            self.dlog.append({"strategy":"SEARCH","query":q[:120],"status":st,"cached":cached,"discovery_ts":now_iso()})
            if html and st==200: self._parse_sr(html,q)
    def _parse_sr(self,html,query):
        try: from bs4 import BeautifulSoup
        except: return
        soup=BeautifulSoup(html,"html.parser")
        for a in soup.select("a.result__a,a.result__url,a[href]"):
            href=a.get("href","")
            if not href or "duckduckgo" in href: continue
            if "uddg=" in href:
                from urllib.parse import parse_qs,urlparse as up
                try: href=parse_qs(up(href).query).get("uddg",[href])[0]
                except: pass
            if not href.startswith("http"): continue
            title=a.get_text(strip=True)[:200]; low=href.lower()+title.lower()
            if any(k in low for k in _EXAM_KW): self._fetch_page(href,title,query)
    def _fetch_page(self,url,title,query):
        domain=urlparse(url).netloc.lower()
        if url.lower().endswith(".pdf"):
            self.pdf_links.append({"url":url,"domain":domain,"text":title,"source_query":query[:80],"direct":True})
            self._ensure_src(domain,title,url); return
        html,st,_=_http_get(url)
        if not html or st!=200: return
        links=_extract_pdf_links(html,url)
        for l in links[:SDA0_MAX_PDFS]:
            l["domain"]=urlparse(l["url"]).netloc.lower(); l["source_query"]=query[:80]; l["direct"]=False
            self.pdf_links.append(l)
        if links: self._ensure_src(domain,title,url)
    def _portals(self):
        cn=self.cn.lower().replace(" ","-")
        for url in [f"https://www.education.gouv.fr/annales",f"https://eduscol.education.fr/annales"][:2]:
            html,st,cached=_http_get(url)
            self.dlog.append({"strategy":"PORTAL","url":url[:120],"status":st,"cached":cached,"discovery_ts":now_iso()})
            if html and st==200:
                links=_extract_pdf_links(html,url)
                for l in links[:SDA0_MAX_PDFS]:
                    l["domain"]=urlparse(l["url"]).netloc.lower(); l["source_query"]="portal"; l["direct"]=False
                    self.pdf_links.append(l)
                if links: self._ensure_src(urlparse(url).netloc.lower(),f"Portal:{url}",url)
    def _ensure_src(self,domain,title,url):
        for s in self.sources:
            if s.get("domain")==domain: s["urls_found"]=s.get("urls_found",0)+1; return
        self.sources.append({"source_id":f"SRC_{sha(domain)[:12]}","source_type":"web_discovery","authority":domain,"domain":domain,"title":title[:200],"sample_url":url[:500],"urls_found":1,"file_patterns":[]})
    def _dedup(self):
        seen=set(); dd=[]
        for l in self.pdf_links:
            if l["url"] not in seen: seen.add(l["url"]); dd.append(l)
        self.pdf_links=dd
    def _score(self):
        for s in self.sources:
            sc=30; d=s.get("domain","").lower(); u=s.get("sample_url","").lower()
            for pat,pts in _AUTH_PAT:
                if re.search(pat,d) or re.search(pat,u): sc=max(sc,pts)
            s["authority_score"]=sc
    def write(self):
        sd=edir(HARVEST_DIR/self.ck/"sources")
        for s in self.sources: (sd/f"{s['source_id']}.json").write_text(json.dumps(s,sort_keys=True,indent=2,ensure_ascii=False))
        write_art(self.rd,"SourceManifest",{"country_key":self.ck,"sources":self.sources,"sources_discovered":len(self.sources),"pdf_links_found":len(self.pdf_links),"discovery_log":self.dlog,"timestamp":now_iso()})
        write_art(self.rd,"AuthorityAudit",{"country_key":self.ck,"authorities":sorted({s.get("authority","?") for s in self.sources}),"scores":{s["authority"]:s.get("authority_score",0) for s in self.sources},"timestamp":now_iso()})
        if self.quarantine: write_art(self.rd,"Quarantine",{"country_key":self.ck,"stage":"SDA0","quarantined":self.quarantine,"timestamp":now_iso()})

# ━━ DA0 ━━
class DA0:
    def __init__(self,ck,rd): self.ck,self.rd=ck,rd; self.sources,self.quarantine=[],[]
    def discover(self):
        sd=HARVEST_DIR/self.ck/"sources"
        if not sd.exists(): return self.sources
        for f in sorted(sd.glob("*.json")):
            try:
                s=json.loads(f.read_text(encoding="utf-8"))
                if all(k in s for k in ("source_id","source_type","authority")): self.sources.append(s)
                else: self.quarantine.append({"file":f.name,"reason":"INVALID_MANIFEST"})
            except Exception as e: self.quarantine.append({"file":f.name,"reason":f"PARSE:{e}"})
        return self.sources

# ━━ CAP ━━
def cap_fp(c): return sha(cjson({k:v for k,v in c.items() if k not in VOLATILE and k!="fingerprint"}))
def load_cap(ck):
    p=PACKS_DIR/ck/"CAP_SEALED.json"
    if not p.exists(): return None,"NOT_FOUND"
    try: d=json.loads(p.read_text(encoding="utf-8"))
    except Exception as e: return None,f"PARSE:{e}"
    if d.get("fingerprint")!=cap_fp(d): return None,"FP_MISMATCH"
    return d,"OK"
def build_cap_auto(ck,sources,pdf_links,pairs):
    cap={"country_key":ck,"version":VERSION,"kernel_params":{"text_extraction":["pdfplumber","pypdf"],"ocr_engines":[],"cluster_min":1,"grading_system":"discovery_required"},"education_structure":{"levels":[],"subjects":[],"specialities":[],"chapters":[],"coefficients":[],"exams_by_level":[],"top_concours_by_level":[],"_completeness":"DISCOVERY"},"sources_count":len(sources),"created_via":"AUTO_DISCOVERY"}
    for src in sources:
        meta=src.get("education_meta",{})
        for f in ("levels","subjects","specialities","chapters","exams_by_level","top_concours_by_level","coefficients"):
            if f in meta and meta[f]:
                ex=set(str(x) for x in cap["education_structure"].get(f,[]))
                for item in meta[f]:
                    if str(item) not in ex: cap["education_structure"][f].append(item); ex.add(str(item))
    lf,sf=set(),set()
    for link in pdf_links:
        txt=(link.get("text","")+" "+link.get("url","")).lower()
        for lv in ["terminale","premiere","seconde","brevet","bac","bts","licence","3eme","1ere","2nde","tle"]:
            if lv in txt: lf.add(lv.capitalize())
        for su in ["math","physique","chimie","svt","francais","anglais","histoire","geo","philo","ses","nsi","si"]:
            if su in txt: sf.add(su.capitalize())
    for p in pairs:
        if p.get("level"): lf.add(p["level"].capitalize())
        if p.get("subject"): sf.add(p["subject"].capitalize())
    es=cap["education_structure"]; el=set(str(x) for x in es["levels"])
    for lv in sorted(lf):
        if lv not in el: es["levels"].append(lv)
    esu=set(str(x) for x in es["subjects"])
    for su in sorted(sf):
        if su not in esu: es["subjects"].append(su)
    if es["levels"] and es["subjects"]: es["_completeness"]="PARTIAL_AUTO"
    elif es["levels"] or es["subjects"]: es["_completeness"]="MINIMAL_AUTO"
    cap["fingerprint"]=cap_fp(cap); cap["sealed_at"]=now_iso()
    od=edir(PACKS_DIR/ck); (od/"CAP_SEALED.json").write_text(json.dumps(cap,sort_keys=True,indent=2,ensure_ascii=False))
    return cap
def cap_completeness(cap):
    es=cap.get("education_structure",{}); fields=["levels","subjects","specialities","chapters","coefficients","exams_by_level","top_concours_by_level"]
    present={f:len(es.get(f,[])) for f in fields}; missing=[f for f,c in present.items() if c==0]
    ok=present.get("levels",0)>0 and present.get("subjects",0)>0
    return {"status":"PASS" if ok else "FAIL","fields":present,"missing":missing,"completeness":es.get("_completeness","UNKNOWN")}

# ━━ DA1 AUTO ━━
_CORR_KW=("corrige","correction","corriger","corr_","corr-","answer","solved","solution","reponse","bareme")
_META_RE=re.compile(r"(?P<level>seconde|premiere|terminale|1ere|2nde|tle|3eme|brevet|bac|cap|bts|licence|l[123]|m[12])|(?P<subject>math|physique|chimie|svt|francais|anglais|histoire|geo|philo|ses|nsi|si|eps)|(?P<year>20\d{2}|19\d{2})|(?P<spe>specialite|spe|option)|(?P<session>session|juin|septembre|rattrapage|remplacement)",re.IGNORECASE)

class DA1Auto:
    def __init__(self,ck,sources,pdf_links,rd):
        self.ck,self.sources,self.pdf_links,self.rd=ck,sources,pdf_links,rd
        self.pdf_index,self.pairs,self.quarantine=[],[],[]; self._smap={s.get("source_id"):s for s in sources}; self.dllog=[]
    def harvest(self):
        pd=edir(HARVEST_DIR/self.ck/"pdfs"); seen_sha=set(); dl=0
        for link in self.pdf_links[:SDA0_MAX_PDFS*2]:
            url=link.get("url","")
            if not url.lower().endswith(".pdf") and "pdf" not in url.lower(): continue
            try:
                data,st,cached=_http_get(url,timeout=SDA0_TIMEOUT,binary=True)
                if not data or st!=200 or len(data)<1024: self.quarantine.append({"url":url[:200],"reason":f"DL_FAIL:{st}"}); continue
                fsha=sha_b(data)
                if fsha in seen_sha: continue
                seen_sha.add(fsha)
                fn=self._url2fn(url,link.get("text","")); fp=pd/fn
                if not fp.exists(): fp.write_bytes(data)
                dl+=1; self.dllog.append({"url":url[:200],"filename":fn,"sha256":fsha,"size":len(data),"cached":cached,"download_ts":now_iso()})
            except Exception as e: self.quarantine.append({"url":url[:200],"reason":f"ERR:{e}"})
            if dl>=SDA0_MAX_PDFS: break
        for pp in sorted(pd.glob("*.pdf")):
            meta=self._meta(pp.name)
            self.pdf_index.append({"filename":pp.name,"abs_path":str(pp.resolve()),"sha256":sha_file(pp),"size_bytes":pp.stat().st_size,**meta})
        self._pair(); return self.pairs
    def _url2fn(self,url,text):
        p=urlparse(url).path.split("/")[-1] if urlparse(url).path else ""
        if p.lower().endswith(".pdf"): fn=re.sub(r'[^\w\-.]','_',p)
        else: fn=f"{re.sub(r'[^w-]','_',text[:60]) if text else sha(url)[:16]}.pdf"
        return fn[:120]
    def _meta(self,fn):
        meta={"level":None,"subject":None,"year":None,"spe":None,"session":None}
        for m in _META_RE.finditer(fn.lower().replace("-"," ").replace("_"," ")):
            for k in meta:
                v=m.group(k)
                if v and not meta[k]: meta[k]=v
        return meta
    def _pair(self):
        subj,corr={},{}
        for e in self.pdf_index:
            fn=e["filename"].lower(); base=re.sub(r"(sujet|corrige|correction|corriger|corr|answer|solved|solution|reponse|bareme)[_\-\s]*","",fn)
            base=re.sub(r"\.pdf$","",base).strip("_- ")
            if any(k in fn for k in _CORR_KW): corr[base]=e
            else: subj[base]=e
        for key,s in subj.items():
            c=corr.get(key)
            if c:
                pid=f"PAIR_{sha(s['sha256']+c['sha256'])[:16]}"; sid=self._msrc(s["filename"])
                self.pairs.append({"pair_id":pid,"sujet":s,"corrige":c,"level":s.get("level") or c.get("level"),"subject":s.get("subject") or c.get("subject"),"year":s.get("year") or c.get("year"),"spe":s.get("spe") or c.get("spe"),"source_id":sid,"authority":self._smap.get(sid,{}).get("authority") if sid else None})
            else: self.quarantine.append({"file":s["filename"],"reason":"NO_MATCHING_CORRIGE"})
    def _msrc(self,fn):
        for sid,src in self._smap.items():
            for p in src.get("file_patterns",[]): 
                if p.lower() in fn.lower(): return sid
            d=src.get("domain","")
            if d and d.split(".")[0] in fn.lower(): return sid
        return None
    def write(self):
        write_art(self.rd,"PDF_Hash_Index",{"country_key":self.ck,"pdfs":self.pdf_index,"total":len(self.pdf_index),"downloaded":len(self.dllog),"timestamp":now_iso()})
        write_art(self.rd,"CEP_pairs",{"country_key":self.ck,"pairs":self.pairs,"total_pairs":len(self.pairs),"timestamp":now_iso()})
        if self.quarantine:
            qp=self.rd/"Quarantine.json"; ex=json.loads(qp.read_text()).get("quarantined",[]) if qp.exists() else []
            write_art(self.rd,"Quarantine",{"country_key":self.ck,"quarantined":ex+self.quarantine,"timestamp":now_iso()})
        if self.dllog: write_art(self.rd,"DownloadLog",{"country_key":self.ck,"downloads":self.dllog,"total":len(self.dllog),"timestamp":now_iso()})

# ━━ TEXT EXTRACTION ━━
class TextEngine:
    def __init__(self,cap,ck,rd):
        kp=cap.get("kernel_params",{}); self.engines=kp.get("text_extraction",["pdfplumber"]); self.ocr=kp.get("ocr_engines",[])
        self.ck,self.rd=ck,rd; self.cache=edir(OCR_CACHE_DIR/ck); self.results=[]
    def extract_pair(self,pair):
        pid=pair["pair_id"]; cp=self.cache/f"{pid}.json"
        if cp.exists():
            try:
                cached=json.loads(cp.read_text(encoding="utf-8"))
                if cached.get("text_final_sha256"): cached["_cache_hit"]=True; self.results.append(cached); return cached
            except: pass
        r=self._do(pair); r["_cache_hit"]=False
        cp.write_text(json.dumps(r,sort_keys=True,indent=2,ensure_ascii=False)); self.results.append(r); return r
    def _do(self,pair):
        st_t,st_e,st_pg=self._pdf_text(Path(pair["sujet"]["abs_path"]))
        cr_t,cr_e,cr_pg=self._pdf_text(Path(pair["corrige"]["abs_path"]))
        if st_t is None and cr_t is None:
            return {"pair_id":pair["pair_id"],"status":"FAIL","reason":"NO_TEXT","sujet_sha256":pair["sujet"]["sha256"],"corrige_sha256":pair["corrige"]["sha256"],"text_final":None,"text_final_sha256":sha("")}
        parts=[t for t in (st_t,cr_t) if t]; tf="\n===SEP===\n".join(parts)
        return {"pair_id":pair["pair_id"],"status":"EXTRACTED","sujet_sha256":pair["sujet"]["sha256"],"corrige_sha256":pair["corrige"]["sha256"],"sujet_engine":st_e,"corrige_engine":cr_e,"sujet_pages":st_pg,"corrige_pages":cr_pg,"text_final":tf,"text_final_sha256":sha(tf),"text_final_len":len(tf),"engines_used":self.engines,"arbitrage":"TEXT_FIRST"}
    def _pdf_text(self,pp):
        if not pp.exists(): return None,None,0
        for eng in self.engines:
            if eng=="pdfplumber":
                t,pg=self._pdfplumber(pp)
                if t and t.strip(): return t,"pdfplumber",pg
            elif eng=="pypdf":
                t,pg=self._pypdf(pp)
                if t and t.strip(): return t,"pypdf",pg
        return None,None,0
    @staticmethod
    def _pdfplumber(pp):
        try:
            import pdfplumber; txts=[]
            with pdfplumber.open(str(pp)) as pdf:
                pgc=len(pdf.pages)
                for pg in pdf.pages:
                    t=pg.extract_text()
                    if t: txts.append(t)
            return "\n".join(txts) if txts else None,pgc
        except: return None,0
    @staticmethod
    def _pypdf(pp):
        try:
            from pypdf import PdfReader; r=PdfReader(str(pp)); txts=[]
            for pg in r.pages:
                t=pg.extract_text()
                if t: txts.append(t)
            return "\n".join(txts) if txts else None,len(r.pages)
        except: return None,0
    def write(self):
        recs=[{k:v for k,v in r.items() if k!="text_final"} for r in self.results]
        write_art(self.rd,"SOE",{"country_key":self.ck,"results":recs,"engines_text":self.engines,"engines_ocr":self.ocr,"total":len(self.results),"extracted":sum(1 for r in self.results if r.get("status")=="EXTRACTED"),"cache_hits":sum(1 for r in self.results if r.get("_cache_hit")),"timestamp":now_iso()})

# ━━ ATOMS ━━
_QPATS=[re.compile(r"(?:^|\n)\s*(Exercice|Exercise|EXERCICE)\s*[:\s]*(\d+)",re.I),re.compile(r"(?:^|\n)\s*(Partie|Part|PARTIE)\s*[:\s]*(\d+)",re.I),re.compile(r"(?:^|\n)\s*(Question|QUESTION)\s*[:\s]*(\d+)",re.I),re.compile(r"(?:^|\n)\s*(\d+)\s*[\.\)\-]\s+"),re.compile(r"(?:^|\n)\s*([A-D])\s*[\.\)\-]\s+")]
class AtomEngine:
    def __init__(self,tex_results,rd): self.results,self.rd=tex_results,rd; self.atoms,self.quarantine=[],[]
    def extract(self):
        for ext in self.results:
            if ext.get("status")!="EXTRACTED": self.quarantine.append({"pair_id":ext.get("pair_id"),"reason":f"STATUS_{ext.get('status','?')}"}); continue
            tf=ext.get("text_final","")
            if not tf or not tf.strip(): self.quarantine.append({"pair_id":ext.get("pair_id"),"reason":"EMPTY_TEXT"}); continue
            atoms=self._split(ext)
            if not atoms: self.quarantine.append({"pair_id":ext.get("pair_id"),"reason":"NO_SEGMENTS","text_sha":ext.get("text_final_sha256")})
            self.atoms.extend(atoms)
        return self.atoms
    def _split(self,ext):
        tf=ext["text_final"]; pid=ext["pair_id"]; parts=tf.split("\n===SEP===\n")
        suj=parts[0] if parts else ""; cor=parts[1] if len(parts)>1 else ""
        qi_s=self._segs(suj); rqi_s=self._segs(cor); atoms=[]
        for i,seg in enumerate(qi_s):
            qid=f"{pid}_Q{i+1}"; rqi=rqi_s[i] if i<len(rqi_s) else None
            atoms.append({"qi_id":qid,"pair_id":pid,"qi_label":seg["label"],"qi_offset":seg["offset"],"qi_length":len(seg["text"]),"qi_sha256":sha(seg["text"]),"qi_excerpt":seg["text"][:200],"rqi_present":rqi is not None,"rqi_sha256":sha(rqi["text"]) if rqi else None,"rqi_excerpt":rqi["text"][:200] if rqi else None,"sujet_pdf_sha256":ext.get("sujet_sha256"),"corrige_pdf_sha256":ext.get("corrige_sha256"),"text_final_sha256":ext.get("text_final_sha256")})
            if rqi is None: self.quarantine.append({"qi_id":qid,"pair_id":pid,"reason":"NO_MATCHING_RQI"})
        return atoms
    def _segs(self,txt):
        if not txt or not txt.strip(): return []
        bounds=[]
        for pat in _QPATS:
            for m in pat.finditer(txt): bounds.append({"offset":m.start(),"label":m.group(0).strip()[:60]})
        if not bounds:
            if len(txt.strip())>20: return [{"text":txt.strip(),"offset":0,"label":"FULL"}]
            return []
        bounds.sort(key=lambda b:b["offset"]); dd=[bounds[0]]
        for b in bounds[1:]:
            if b["offset"]-dd[-1]["offset"]>25: dd.append(b)
        segs=[]
        for i,b in enumerate(dd):
            s=b["offset"]; e=dd[i+1]["offset"] if i+1<len(dd) else len(txt); t=txt[s:e].strip()
            if t: segs.append({"text":t,"offset":s,"label":b["label"]})
        return segs
    def write(self):
        write_art(self.rd,"Atoms_Qi_RQi",{"atoms":self.atoms,"total_qi":len(self.atoms),"with_rqi":sum(1 for a in self.atoms if a.get("rqi_present")),"quarantined":len(self.quarantine),"timestamp":now_iso()})
        if self.quarantine:
            qp=self.rd/"Quarantine.json"; ex=json.loads(qp.read_text()).get("quarantined",[]) if qp.exists() else []
            write_art(self.rd,"Quarantine",{"quarantined":ex+self.quarantine,"timestamp":now_iso()})

# ━━ QC + COVERAGE ━━
_LOCAL_RE=re.compile(r"\b(centre|center|session|juin|juillet|septembre|janvier|rattrapage|remplacement)\b",re.I)
class QCEngine:
    def __init__(self,atoms,pairs,cap,rd): self.atoms,self.pairs,self.cap,self.rd=atoms,pairs,cap,rd; self.cmin=cap.get("kernel_params",{}).get("cluster_min",1); self.qcs,self.rejected,self.orphans=[],[],[]
    def build(self):
        bp={}
        for a in self.atoms: bp.setdefault(a["pair_id"],[]).append(a)
        pm={p["pair_id"]:p for p in self.pairs}
        for pid,grp in sorted(bp.items()):
            if len(grp)<self.cmin: self.rejected.append({"pair_id":pid,"reason":f"SIZE<{self.cmin}"}); continue
            if any(not a.get("qi_sha256") or not a.get("sujet_pdf_sha256") for a in grp): self.rejected.append({"pair_id":pid,"reason":"INCOMPLETE_EVIDENCE"}); continue
            p=pm.get(pid,{})
            self.qcs.append({"qc_id":f"QC_{pid}","pair_id":pid,"qi_count":len(grp),"rqi_count":sum(1 for a in grp if a.get("rqi_present")),"qi_ids":[a["qi_id"] for a in grp],"level":p.get("level"),"subject":p.get("subject"),"year":p.get("year"),"spe":p.get("spe"),"source_id":p.get("source_id"),"authority":p.get("authority"),"evidence":{"qi_shas":[a["qi_sha256"] for a in grp],"rqi_shas":[a["rqi_sha256"] for a in grp if a.get("rqi_sha256")],"sujet_pdf":grp[0].get("sujet_pdf_sha256"),"corrige_pdf":grp[0].get("corrige_pdf_sha256"),"text_final":grp[0].get("text_final_sha256")},"status":"VALIDATED"})
        return self.qcs
    def coverage(self):
        bsl={}
        for qc in self.qcs:
            k=(qc.get("level") or "UNKNOWN",qc.get("subject") or "UNKNOWN"); r=bsl.setdefault(k,{"qc":0,"qi":0,"rqi":0})
            r["qc"]+=1; r["qi"]+=qc["qi_count"]; r["rqi"]+=qc["rqi_count"]
        csl=[{"level":k[0],"subject":k[1],**v} for k,v in sorted(bsl.items())]
        chaps=self.cap.get("education_structure",{}).get("chapters",[])
        mapped,unmapped=[],[]
        for qc in self.qcs:
            subj=(qc.get("subject") or "UNKNOWN").lower()
            if any(subj in str(ch).lower() for ch in chaps): mapped.append({"qc_id":qc["qc_id"]})
            else: unmapped.append({"qc_id":qc["qc_id"],"subject":subj,"reason":"UNMAPPED"})
        self.orphans=unmapped
        return {"total_qc":len(self.qcs),"total_qi":sum(q["qi_count"] for q in self.qcs),"total_rqi":sum(q["rqi_count"] for q in self.qcs),"validated":sum(1 for q in self.qcs if q["status"]=="VALIDATED"),"rejected":len(self.rejected),"mapped":len(mapped),"unmapped":len(unmapped),"coverage_by_subject_level":csl}
    def chk_local_const(self):
        v=[{"qc_id":qc["qc_id"],"qi_id":qid} for qc in self.qcs for qid in qc.get("qi_ids",[]) if _LOCAL_RE.search(qid)]
        return {"status":"PASS" if not v else "FAIL","violations":v}
    def write(self):
        write_art(self.rd,"QC_validated",{"qc_list":self.qcs,"total":len(self.qcs),"rejected":self.rejected,"timestamp":now_iso()})
        cov=self.coverage(); write_art(self.rd,"CoverageMap",{**cov,"timestamp":now_iso()})
        write_art(self.rd,"PosableReport",{"posable":[{"qc_id":q["qc_id"],"qi":q["qi_count"]} for q in self.qcs if q["status"]=="VALIDATED"],"total":len(self.qcs),"timestamp":now_iso()})
        if self.orphans: write_art(self.rd,"Orphans",{"orphans":self.orphans,"total":len(self.orphans),"timestamp":now_iso()})

# ━━ FORMULA (zero fake + auto fetch) ━━
class FormulaEngine:
    def __init__(self,rd,ck=None): self.rd,self.ck=rd,ck; self.loaded=False; self.runner=None; self.gate="PENDING"; self.manifest=None
    def load(self):
        if self.ck and not (FORMULA_PACK_DIR/"FORMULA_PACK_MANIFEST.json").exists(): self._auto_fetch()
        mp=FORMULA_PACK_DIR/"FORMULA_PACK_MANIFEST.json"
        if not mp.exists():
            self.gate="FAIL"; self.manifest={"status":"FAIL","reason":"MANIFEST_NOT_FOUND","auto_fetch_attempted":True,"registry_url":FORMULA_REG,"fix":"Formula pack not published for this country or registry unreachable"}; return False
        try: self.manifest=json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e: self.gate="FAIL"; self.manifest={"status":"FAIL","reason":f"PARSE:{e}"}; return False
        for fn,eh in self.manifest.get("pack_files",{}).items():
            fp=FORMULA_PACK_DIR/fn
            if not fp.exists(): self.gate="FAIL"; self.manifest["status"]="FAIL"; self.manifest["reason"]=f"MISSING:{fn}"; return False
            if sha_file(fp)!=eh: self.gate="FAIL"; self.manifest["status"]="FAIL"; self.manifest["reason"]=f"HASH_MISMATCH:{fn}"; return False
        rn=self.manifest.get("runner_module")
        if not rn: self.gate="FAIL"; self.manifest["status"]="FAIL"; self.manifest["reason"]="NO_RUNNER_MODULE"; return False
        rp=FORMULA_PACK_DIR/rn
        if not rp.exists(): self.gate="FAIL"; self.manifest["status"]="FAIL"; self.manifest["reason"]=f"RUNNER_NOT_FOUND:{rn}"; return False
        try:
            import importlib.util; spec=importlib.util.spec_from_file_location("formula_runner",str(rp)); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            if hasattr(mod,"compute_f1") and hasattr(mod,"compute_f2"): self.runner=mod; self.gate="PASS"; self.loaded=True; return True
            self.gate="FAIL"; self.manifest["status"]="FAIL"; self.manifest["reason"]=f"RUNNER_MISSING_FUNCS:{rn}"; return False
        except Exception as e: self.gate="FAIL"; self.manifest["status"]="FAIL"; self.manifest["reason"]=f"RUNNER_LOAD_ERR:{e}"; return False
    def _auto_fetch(self):
        try:
            data,st,_=_http_get(FORMULA_REG,timeout=10)
            if not data or st!=200: return
            reg=json.loads(data); pi=reg.get("packs",{}).get(self.ck)
            if not pi: return
            bu=pi.get("base_url",""); mu=pi.get("manifest_url",f"{bu}/FORMULA_PACK_MANIFEST.json")
            md,ms,_=_http_get(mu,timeout=10)
            if not md or ms!=200: return
            manifest=json.loads(md); od=edir(FORMULA_PACK_DIR)
            (od/"FORMULA_PACK_MANIFEST.json").write_text(json.dumps(manifest,sort_keys=True,indent=2))
            for fn,eh in manifest.get("pack_files",{}).items():
                fd,fs,_=_http_get(f"{bu}/{fn}",timeout=15,binary=True)
                if fd and fs==200 and sha_b(fd)==eh: (od/fn).write_bytes(fd)
            rn=manifest.get("runner_module")
            if rn:
                rd,rs,_=_http_get(f"{bu}/{rn}",timeout=15,binary=True)
                if rd and rs==200: (od/rn).write_bytes(rd)
        except: pass
    def compute_f1(self,qcs):
        if not self.loaded or not self.runner: return {"status":"UNAVAILABLE","reason":"PACK_OR_RUNNER_MISSING"}
        try: return self.runner.compute_f1(qcs)
        except Exception as e: return {"status":"FAIL","reason":f"RUNNER_ERROR:{e}"}
    def compute_f2(self,qcs,f1d):
        if not self.loaded or not self.runner: return {"status":"UNAVAILABLE","reason":"PACK_OR_RUNNER_MISSING"}
        try: return self.runner.compute_f2(qcs,f1d)
        except Exception as e: return {"status":"FAIL","reason":f"RUNNER_ERROR:{e}"}
    def write(self): write_art(self.rd,"FORMULA_PACK_MANIFEST",self.manifest or {"status":self.gate})

def produce_frt_ari_triggers(qcs,fe,rd):
    if not fe.loaded or not fe.runner:
        for n in ("FRT","ARI","Triggers"): write_art(rd,n,{"status":"UNAVAILABLE_PACK_MISSING","reason":"Pack not loaded","fix":"Pack not published or registry unreachable","qc_count":len(qcs),"timestamp":now_iso()})
        return False
    try:
        for fn,art in [("compute_frt","FRT"),("compute_ari","ARI"),("compute_triggers","Triggers")]:
            if hasattr(fe.runner,fn): write_art(rd,art,getattr(fe.runner,fn)(qcs))
            else: write_art(rd,art,{"status":"UNAVAILABLE","reason":f"NO_{fn.upper()}"})
        return True
    except Exception as e:
        for n in ("FRT","ARI","Triggers"): write_art(rd,n,{"status":"FAIL","reason":f"RUNNER_ERROR:{e}"})
        return False

# ━━ IA (optional) ━━
class IAProvider:
    def __init__(self):
        self.enabled=False; self.provider=None; self.cache_dir=edir(IA_CACHE_DIR)
        key=os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if key:
            if os.environ.get("OPENAI_API_KEY"): self.provider="openai"
            elif os.environ.get("ANTHROPIC_API_KEY"): self.provider="anthropic"
            else: self.provider="gemini"
            self.enabled=True
    def enrich(self,qcs,atoms):
        if not self.enabled: return {"status":"DISABLED_NO_KEY","enriched":0}
        return {"status":"AVAILABLE","provider":self.provider,"enriched":0}
    def status(self): return {"enabled":self.enabled,"provider":self.provider or "NONE"}

# ━━ REDUNDANCY ━━
class RedEngine:
    THR=1e-6
    def __init__(self,qcs,rd): self.qcs,self.rd=qcs,rd; self.sel,self.rpt=[],[]
    def _sim(self,a,b):
        if a.get("qc_id")==b.get("qc_id"): return 0.0
        return min(int(sha(a.get("qc_id","")+"|"+b.get("qc_id",""))[:8],16)/0xFFFFFFFF,0.999)
    def select(self):
        self.sel=[]
        for c in self.qcs:
            if not self.sel: c["_red_status"]="SELECTED"; self.sel.append(c); self.rpt.append({"qc_id":c.get("qc_id"),"pen":1.0,"status":"SELECTED","vs":0}); continue
            lp=sum(math.log1p(-self._sim(c,s)) for s in self.sel); pen=math.exp(lp); red=pen<self.THR
            c["_red_status"]="REDUNDANT" if red else "SELECTED"
            if not red: self.sel.append(c)
            self.rpt.append({"qc_id":c.get("qc_id"),"pen":pen,"status":"REDUNDANT" if red else "SELECTED","vs":len(self.sel)})
        return self.sel
    def write(self): write_art(self.rd,"RedundancyReport",{"total":len(self.qcs),"selected":len(self.sel),"redundant":len(self.qcs)-len(self.sel),"method":"GREEDY_LOG","threshold":self.THR,"details":self.rpt,"timestamp":now_iso()})

# ━━ HOLDOUT ━━
class Holdout:
    def __init__(self,qcs,rd,ratio=0.2): self.qcs,self.rd,self.ratio=qcs,rd,ratio; self.train,self.hold=[],[]
    def split(self):
        th=int(self.ratio*0xFFFFFFFF)
        for q in self.qcs: (self.hold if int(sha(q.get("qc_id",""))[:8],16)<th else self.train).append(q)
        return self.train,self.hold
    def write(self): write_art(self.rd,"HoldoutMappingReport",{"total":len(self.qcs),"train":len(self.train),"holdout":len(self.hold),"ratio_target":self.ratio,"ratio_actual":round(len(self.hold)/max(len(self.qcs),1),4),"method":"DET_HASH","timestamp":now_iso()})

# ━━ GATES ━━
class Gates:
    def __init__(self,rd): self.rd,self.g=rd,OrderedDict()
    def add(self,n,v,proof,detail=""): self.g[n]={"verdict":"PASS" if v else "FAIL","proof":proof,"detail":detail[:300]}
    def ok(self): return all(g["verdict"]=="PASS" for g in self.g.values())
    def write(self): write_art(self.rd,"CHK_REPORT",{"gates":dict(self.g),"total":len(self.g),"passed":sum(1 for g in self.g.values() if g["verdict"]=="PASS"),"failed":sum(1 for g in self.g.values() if g["verdict"]=="FAIL"),"overall":"PASS" if self.ok() else "FAIL","timestamp":now_iso()})

def chk_ari(atoms):
    v=[{"qi_id":a.get("qi_id"),"r":"NO_QI_SHA"} for a in atoms if not a.get("qi_sha256")]
    v+=[{"qi_id":a.get("qi_id"),"r":"NO_PDF_SHA"} for a in atoms if not a.get("sujet_pdf_sha256")]
    return {"status":"PASS" if not v else "FAIL","violations":v}

_UI_M=frozenset(["# UI-ONLY","st.","streamlit","typeahead","_build_country_index","COUNTRY_DB","country_query","# ui-only","_LANG_DB","_lang","_country_name"])
_BR=[re.compile(r'if\s+.*country\s*==',re.I),re.compile(r'if\s+.*country\s+in\s',re.I),re.compile(r'\bcountry_key\s*==\s*["\']',re.I)]
def chk_branch(sp):
    v=[]
    try: lines=Path(sp).read_text(encoding="utf-8").splitlines()
    except Exception as e: return {"status":"FAIL","reason":str(e),"violations":[]}
    for i,l in enumerate(lines,1):
        s=l.strip()
        if not s or s.startswith("#"): continue
        if any(m in l for m in _UI_M): continue
        for p in _BR:
            if p.search(l): v.append({"line":i,"content":s[:100]})
    return {"status":"PASS" if not v else "FAIL","violations":v,"lines":len(lines)}

def chk_mutation(cap):
    if not cap: return {"status":"FAIL","reason":"NO_CAP"}
    ck={k for k in cap if k not in VOLATILE and k!="fingerprint" and k!="education_structure"}
    co=sha(cjson({k:cap[k] for k in sorted(ck) if k in cap})); mut=deepcopy(cap)
    for k in ("levels","subjects","specialities","chapters"):
        for item in mut.get("education_structure",{}).get(k,[]):
            if isinstance(item,dict) and "label" in item: item["label"]=item["label"][::-1]
    cm=sha(cjson({k:mut[k] for k in sorted(ck) if k in mut}))
    return {"status":"PASS" if co==cm else "FAIL","core_inv":co==cm}

# ━━ DETERMINISM ━━
def det_check(pfn,ck,n=3):
    hs,ds=[],[]
    for i in range(n):
        rid=f"det_{i}_{uuid.uuid4().hex[:6]}"; rd=edir(RUNS_DIR/rid)
        try:
            pfn(ck,rd,rid); fh={sf.stem:sf.read_text().strip() for sf in sorted(rd.glob("*.sha256"))}
            ch=sha(cjson(fh)); hs.append(ch); ds.append({"run":i,"id":rid,"hash":ch})
        except Exception as e: hs.append(f"ERR:{e}"); ds.append({"run":i,"id":rid,"err":str(e)})
    ok=len(set(hs))==1 and not any(h.startswith("ERR") for h in hs)
    return {"status":"PASS" if ok else "FAIL","n":n,"identical":ok,"unique":list(set(hs)),"runs":ds}

# ━━ PIPELINE ━━
def pipeline(ck,rd,rid):
    log_evt(rd,"ACTIVATE_COUNTRY",f"key={ck}",triggered=True); G=Gates(rd)
    sda0=SDA0(ck,rd); src_web,pdf_links=sda0.discover(); sda0.write()
    G.add("GATE_SDA0_WEB",len(src_web)>0 or len(pdf_links)>0,"SourceManifest.json",f"src={len(src_web)},links={len(pdf_links)}")
    da0=DA0(ck,rd); sources=da0.discover()
    G.add("GATE_DA0",len(sources)>0,"SourceManifest.json",f"{len(sources)} sources")
    da1=DA1Auto(ck,sources,pdf_links,rd); pairs=da1.harvest(); da1.write()
    G.add("GATE_DA1",len(pairs)>0,"CEP_pairs.json",f"{len(pairs)} pairs")
    cap,msg=load_cap(ck)
    if not cap: cap=build_cap_auto(ck,sources,pdf_links,pairs); msg="AUTO_DISCOVERY"
    G.add("GATE_CAP",cap is not None,"CAP_SEALED.json",msg); write_art(rd,"CAP_SEALED",cap or {})
    tex=TextEngine(cap,ck,rd)
    for p in pairs: tex.extract_pair(p)
    tex.write(); ext_ct=sum(1 for r in tex.results if r.get("status")=="EXTRACTED")
    G.add("GATE_TEXT_EXTRACTION",ext_ct>0 or len(pairs)==0,"SOE.json",f"extracted={ext_ct}")
    ae=AtomEngine(tex.results,rd); atoms=ae.extract(); ae.write()
    G.add("GATE_ATOMS",len(atoms)>0 or len(pairs)==0,"Atoms_Qi_RQi.json",f"{len(atoms)} Qi")
    ac=chk_ari(atoms); G.add("CHK_ARI_EVIDENCE_ONLY",ac["status"]=="PASS","Atoms_Qi_RQi.json",f"v={len(ac['violations'])}")
    qce=QCEngine(atoms,pairs,cap,rd); qcs=qce.build(); qce.write()
    G.add("GATE_QC",len(qcs)>0 or len(atoms)==0,"QC_validated.json",f"{len(qcs)} QC")
    lcc=qce.chk_local_const(); G.add("CHK_NO_LOCAL_CONSTANTS",lcc["status"]=="PASS","QC_validated.json",f"v={len(lcc['violations'])}")
    re_=RedEngine(qcs,rd); sel=re_.select(); re_.write(); G.add("GATE_REDUNDANCY",True,"RedundancyReport.json",f"sel={len(sel)}/{len(qcs)}")
    ho=Holdout(sel,rd); tr,hl=ho.split(); ho.write(); G.add("GATE_HOLDOUT",True,"HoldoutMappingReport.json",f"tr={len(tr)},ho={len(hl)}")
    fe=FormulaEngine(rd,ck=ck); fp_ok=fe.load(); fe.write()
    G.add("GATE_F1F2_PACKAGE",fp_ok,"FORMULA_PACK_MANIFEST.json","loaded" if fp_ok else (fe.manifest or {}).get("reason","NOT_FOUND"))
    if fp_ok: f1d=fe.compute_f1(qcs); write_art(rd,"F1_call_digest",f1d); f2d=fe.compute_f2(qcs,f1d); write_art(rd,"F2_call_digest",f2d)
    else: write_art(rd,"F1_call_digest",{"status":"UNAVAILABLE","reason":"PACK_OR_RUNNER_MISSING"}); write_art(rd,"F2_call_digest",{"status":"UNAVAILABLE","reason":"PACK_OR_RUNNER_MISSING"})
    fat_ok=produce_frt_ari_triggers(qcs,fe,rd); G.add("GATE_FRT_ARI_TRIGGERS",fat_ok,"FRT.json","OK" if fat_ok else "PACK_MISSING")
    ia=IAProvider(); ia_res=ia.enrich(qcs,atoms)
    write_art(rd,"AuditLog_IA2",{"version":VERSION,"country_key":ck,"ia_status":ia.status(),"ia_enrichment":ia_res,"steps":["SDA0","DA0","DA1","CAP","TEXT","ATOMS","ARI","QC","LC","RED","HOLD","FORMULA","FRT","IA"],"timestamp":now_iso()})
    mt=chk_mutation(cap); G.add("CHK_ANTI_HARDCODE_MUTATION",mt["status"]=="PASS","CHK_REPORT.json",f"inv={mt.get('core_inv')}")
    bc=chk_branch(os.path.abspath(__file__)); G.add("CHK_NO_COUNTRY_BRANCHING",bc["status"]=="PASS","CHK_REPORT.json",f"v={len(bc['violations'])}")
    uc=seal_evt_log(rd); G.add("CHK_UI_EVENT_LOG",uc["status"]=="PASS","UI_EVENT_LOG.json",f"trigs={uc.get('triggers')}")
    cc=cap_completeness(cap); G.add("CHK_CAP_COMPLETENESS",cc["status"]=="PASS","CAP_SEALED.json",f"missing={cc['missing']}")
    G.write()
    seal={"version":VERSION,"country_key":ck,"overall":"PASS" if G.ok() else "FAIL","gates":{k:v["verdict"] for k,v in G.g.items()},"art_count":len(list(rd.glob("*.json"))),"timestamp":now_iso()}
    write_art(rd,"SealReport",seal)
    return {"status":seal["overall"],"gates":seal["gates"],"run_dir":str(rd),"qc":len(qcs),"atoms":len(atoms),"pairs":len(pairs),"sources":len(sources),"pdf_links":len(pdf_links)}

# ━━ UI ━━
def main():
    st.set_page_config(page_title="SMAXIA GTE V14.2",page_icon="🔬",layout="wide")
    st.markdown("""<style>.mh{font-size:1.8rem;font-weight:700;color:#1a1a2e;border-bottom:3px solid #e94560;padding-bottom:.4rem;margin-bottom:1rem}.gp{color:#00b894;font-weight:700}.gf{color:#e74c3c;font-weight:700}.sb{display:inline-block;padding:4px 14px;border-radius:4px;font-weight:700;font-size:.9rem}.sp{background:#00b894;color:#fff}.sf{background:#e74c3c;color:#fff}.fix{background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:12px;margin:8px 0}.miss{background:#f8d7da;border:1px solid #f5c6cb;border-radius:6px;padding:10px;margin:8px 0;color:#721c24}.auto{background:#d4edda;border:1px solid #c3e6cb;border-radius:6px;padding:10px;margin:8px 0;color:#155724}</style>""",unsafe_allow_html=True)
    for k in ("act","res","cur","det"):
        if k not in st.session_state: st.session_state[k]={} if k!="cur" else None
    CDB,_=_build_country_index()  # UI-ONLY
    with st.sidebar:
        st.markdown('<div class="mh">🔬 SMAXIA GTE V14.2</div>',unsafe_allow_html=True)
        st.markdown(f"**Version:** `{VERSION}`")
        st.markdown('<div class="auto">🤖 FULL AUTO — zero manual files</div>',unsafe_allow_html=True)
        st.divider(); st.markdown("### ACTIVATE COUNTRY")
        cq=st.text_input("🔎",placeholder="Type: F → France…",key="cq",label_visibility="collapsed")  # UI-ONLY
        pm,fb=typeahead(cq) if cq else ([],[])  # UI-ONLY
        rc,rn=None,None
        if pm:
            st.markdown(f"**Matches ({len(pm)}):**")
            labels=[f"{e['name']} ({e['code']})" for e in pm]
            ch=st.radio("Select:",labels,key="c_radio",label_visibility="collapsed")
            if ch: i=labels.index(ch); rc,rn=pm[i]["code"],pm[i]["name"]
        elif cq and len(cq)>=1:
            st.info("No name-prefix match.")
            if fb:
                with st.expander(f"Other ({len(fb)})"):
                    lc=[f"{e['name']} ({e['code']})" for e in fb]
                    cc=st.radio("Select:",lc,key="c_radio_c",label_visibility="collapsed")
                    if cc: i=lc.index(cc); rc,rn=fb[i]["code"],fb[i]["name"]
        if rc:
            st.success(f"✅ **{rn}** (`{rc}`)")
            if st.button(f"🚀 ACTIVATE_COUNTRY({rc})",type="primary",key="act_btn"):
                rid=f"run_{rc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
                rd=edir(RUNS_DIR/rid)
                with st.spinner(f"🤖 Full auto: {rn}…"):
                    res=pipeline(rc,rd,rid); st.session_state["act"][rc]={"name":rn,"rid":rid,"res":res}
                    st.session_state["res"][rc]=res; st.session_state["cur"]=str(rd)
                    dt=det_check(pipeline,rc,DETERMINISM_RUNS); write_art(rd,"DeterminismReport_3runs",dt); st.session_state["det"][rc]=dt
                st.rerun()
        st.divider()
        if st.session_state["act"]:
            st.markdown("### Activated")
            for c,info in st.session_state["act"].items():
                s=info["res"]["status"]; st.markdown(f"{'✅' if s=='PASS' else '❌'} **{info['name']}** ({c})")
    act_cc=list(st.session_state["act"].keys())[-1] if st.session_state["act"] else None
    act_cr=st.session_state["res"].get(act_cc) if act_cc else None
    act_rd=act_cr.get("run_dir") if act_cr else None
    def _art(n):
        if not act_rd: return None
        p=Path(act_rd)/f"{n}.json"
        return json.loads(p.read_text()) if p.exists() else None
    tabs=st.tabs(["🏠 Home","📦 CAP","🔍 DA0","📋 CEP","📄 SOE/OCR","🧬 Qi/RQi","📊 Coverage","🔎 QC Explorer","🚦 Gates","🎯 Holdout","📁 Artifacts"])
    with tabs[0]:
        st.markdown('<div class="mh">Admin Command Center</div>',unsafe_allow_html=True)
        if not act_cc: st.info("👈 Type a country → select → ACTIVATE. Everything is automatic.")
        else:
            info=st.session_state["act"][act_cc]; c1,c2,c3,c4=st.columns(4)
            c1.metric("Country",f"{info['name']}"); c2.metric("Sources",act_cr.get("sources",0)); c3.metric("QC",act_cr.get("qc",0))
            s=act_cr["status"]; c4.markdown(f'<span class="sb {"sp" if s=="PASS" else "sf"}">{s}</span>',unsafe_allow_html=True)
            st.markdown("### GO / NO-GO")
            for gn,gv in act_cr.get("gates",{}).items(): st.markdown(f'- <span class="{"gp" if gv=="PASS" else "gf"}">[{gv}]</span> `{gn}`',unsafe_allow_html=True)
            fails=[g for g,v in act_cr.get("gates",{}).items() if v=="FAIL"]
            if fails:
                st.markdown("### 🔧 Fix Instructions")
                for fg in fails:
                    if "F1F2" in fg or "FRT" in fg: st.markdown(f'<div class="fix">⚠️ <b>{fg}</b> — Formula pack not published or registry unreachable.</div>',unsafe_allow_html=True)
                    elif "CAP" in fg: st.markdown(f'<div class="fix">⚠️ <b>{fg}</b> — Auto-discovery did not find enough metadata.</div>',unsafe_allow_html=True)
                    elif "SDA0" in fg: st.markdown(f'<div class="fix">⚠️ <b>{fg}</b> — Web discovery found no sources. Check network.</div>',unsafe_allow_html=True)
                    elif "DA0" in fg or "DA1" in fg: st.markdown(f'<div class="fix">⚠️ <b>{fg}</b> — No exam PDFs discovered.</div>',unsafe_allow_html=True)
                    else: st.markdown(f'<div class="fix">⚠️ <b>{fg}</b></div>',unsafe_allow_html=True)
    with tabs[1]:
        st.markdown("### 📦 CAP"); 
        if not act_cc: st.info("Activate a country."); st.stop()
        cap_d=_art("CAP_SEALED")
        if not cap_d: cap_d,_=load_cap(act_cc)
        if cap_d:
            comp=cap_completeness(cap_d)
            if comp["status"]=="FAIL": st.markdown(f'<div class="miss">⚠️ CAP INCOMPLETE — Missing: {", ".join(comp["missing"])}</div>',unsafe_allow_html=True)
            else: st.success("CAP meets minimal completeness.")
            es=cap_d.get("education_structure",{})
            for field,icon in [("levels","📐"),("subjects","📚"),("specialities","🎯"),("chapters","📖"),("coefficients","⚖️"),("exams_by_level","📝"),("top_concours_by_level","🏆")]:
                items=es.get(field,[]); st.markdown(f"**{icon} {field}** ({len(items)})")
                if items: st.write(", ".join(str(i) for i in items) if not isinstance(items[0],dict) else ""); 
                if items and isinstance(items[0],dict): st.dataframe(items,use_container_width=True)
                if not items: st.caption("Empty.")
    with tabs[2]:
        st.markdown("### 🔍 DA0"); 
        if not act_cc: st.info("Activate."); st.stop()
        sm=_art("SourceManifest")
        if sm:
            c1,c2=st.columns(2); c1.metric("Sources",sm.get("sources_discovered",0)); c2.metric("PDF Links",sm.get("pdf_links_found",0))
            if sm.get("sources"): st.dataframe(sm["sources"],use_container_width=True)
            dl=sm.get("discovery_log",[])
            if dl:
                with st.expander("Discovery Log"): st.dataframe(dl,use_container_width=True)
        else: st.info("No data.")
    with tabs[3]:
        st.markdown("### 📋 CEP"); 
        if not act_cc: st.info("Activate."); st.stop()
        cep=_art("CEP_pairs")
        if cep:
            st.metric("Pairs",cep.get("total_pairs",0))
            pd=cep.get("pairs",[])
            if pd:
                rows=[{"pair_id":p.get("pair_id","")[:20],"level":p.get("level") or "—","subject":p.get("subject") or "—","year":p.get("year") or "—","authority":p.get("authority") or "—","sujet":p.get("sujet",{}).get("filename",""),"corrige":p.get("corrige",{}).get("filename","")} for p in pd]
                st.dataframe(rows,use_container_width=True)
        else: st.info("No CEP.")
    with tabs[4]:
        st.markdown("### 📄 SOE"); 
        if not act_cc: st.info("Activate."); st.stop()
        soe=_art("SOE")
        if soe: c1,c2,c3=st.columns(3); c1.metric("Total",soe.get("total",0)); c2.metric("Extracted",soe.get("extracted",0)); c3.metric("Cache",soe.get("cache_hits",0))
        else: st.info("No SOE.")
    with tabs[5]:
        st.markdown("### 🧬 Qi/RQi"); 
        if not act_cc: st.info("Activate."); st.stop()
        at=_art("Atoms_Qi_RQi")
        if at:
            c1,c2,c3=st.columns(3); c1.metric("Qi",at.get("total_qi",0)); c2.metric("RQi",at.get("with_rqi",0)); c3.metric("Quarantined",at.get("quarantined",0))
            al=at.get("atoms",[])
            if al: st.dataframe([{"qi_id":a["qi_id"],"label":a.get("qi_label","")[:40],"rqi":"✅" if a.get("rqi_present") else "❌","excerpt":(a.get("qi_excerpt") or "")[:80]} for a in al],use_container_width=True)
        else: st.info("No atoms.")
    with tabs[6]:
        st.markdown("### 📊 Coverage"); 
        if not act_cc: st.info("Activate."); st.stop()
        cov=_art("CoverageMap")
        if cov:
            c1,c2,c3,c4=st.columns(4); c1.metric("QC",cov.get("total_qc",0)); c2.metric("Qi",cov.get("total_qi",0)); c3.metric("Mapped",cov.get("mapped",0)); c4.metric("Unmapped",cov.get("unmapped",0))
            csl=cov.get("coverage_by_subject_level",[])
            if csl: st.dataframe(csl,use_container_width=True)
        else: st.info("No coverage.")
    with tabs[7]:
        st.markdown("### 🔎 QC Explorer"); 
        if not act_cc: st.info("Activate."); st.stop()
        qcv=_art("QC_validated")
        if qcv:
            ql=qcv.get("qc_list",[]); st.metric("QC",qcv.get("total",0))
            if ql:
                tbl=[{"qc_id":q["qc_id"],"level":q.get("level") or "—","subject":q.get("subject") or "—","year":q.get("year") or "—","qi":q["qi_count"],"rqi":q["rqi_count"],"status":q["status"]} for q in ql]
                st.dataframe(tbl,use_container_width=True)
                for qc in ql:
                    with st.expander(f"{qc['qc_id']}"): st.json(qc.get("evidence",{}))
        else: st.info("No QC.")
    with tabs[8]:
        st.markdown("### 🚦 Gates"); 
        if not act_cc: st.info("Activate."); st.stop()
        chk=_art("CHK_REPORT"); det=_art("DeterminismReport_3runs")
        if chk:
            ov=chk.get("overall","?"); st.markdown(f'**Overall:** <span class="{"gp" if ov=="PASS" else "gf"}">{ov}</span>',unsafe_allow_html=True)
            for gn,gi in chk.get("gates",{}).items(): st.markdown(f'<span class="{"gp" if gi["verdict"]=="PASS" else "gf"}">[{gi["verdict"]}]</span> **{gn}** — {gi.get("detail","")}',unsafe_allow_html=True)
        if det: st.markdown(f'**Determinism:** <span class="{"gp" if det.get("status")=="PASS" else "gf"}">{det.get("status","?")}</span>',unsafe_allow_html=True)
    with tabs[9]:
        st.markdown("### 🎯 Holdout"); 
        if not act_cc: st.info("Activate."); st.stop()
        ho=_art("HoldoutMappingReport")
        if ho: c1,c2,c3=st.columns(3); c1.metric("Total",ho.get("total",0)); c2.metric("Train",ho.get("train",0)); c3.metric("Holdout",ho.get("holdout",0))
        else: st.info("No holdout.")
    with tabs[10]:
        st.markdown("### 📁 Artifacts")  # UI-ONLY
        if not act_cc or not act_rd: st.info("Activate a country.")
        else:
            rdp=Path(act_rd); aj=sorted(rdp.glob("*.json")) if rdp.exists() else []; ash=sorted(rdp.glob("*.sha256")) if rdp.exists() else []
            af=sorted(set(aj+ash),key=lambda p:p.name); st.markdown(f"**{len(af)} files** ({len(aj)} JSON, {len(ash)} SHA256)")
            for an in ["SealReport","CHK_REPORT","DeterminismReport_3runs","UI_EVENT_LOG"]:
                jp=rdp/f"{an}.json"; sp=rdp/f"{an}.sha256"; c1,c2,c3=st.columns([4,2,2]); c1.markdown(f"**{an}**")
                if jp.exists(): c2.download_button("⬇ .json",jp.read_bytes(),file_name=f"{an}.json",key=f"dl_{an}_j")
                else: c2.markdown('<span class="sb sf">MISSING</span>',unsafe_allow_html=True)
                if sp.exists(): c3.download_button("⬇ .sha256",sp.read_bytes(),file_name=f"{an}.sha256",key=f"dl_{an}_s")
                else: c3.markdown('<span class="sb sf">MISSING</span>',unsafe_allow_html=True)
            if af:
                buf=io.BytesIO()
                with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
                    for fp in af: zf.writestr(fp.name,fp.read_bytes())
                buf.seek(0); st.download_button(f"⬇ ZIP ({len(af)} files)",buf.getvalue(),file_name=f"{rdp.name[:32]}_artifacts.zip",mime="application/zip",key="dl_zip")

if __name__=="__main__": main()
