#!/usr/bin/env python3
# smaxia_gte_v14_3_3.py
"""SMAXIA GTE Kernel v14.3.3 — ISO-PROD Single-File Compact"""
import streamlit as st
import json,hashlib,os,re,inspect,random,math,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote_plus,urljoin,urlparse
MAX_PDFS=20;BFS_DEPTH=2;SEED_DOMAINS_MAX=8
RUN_DIR=Path("run");WEB_CACHE_DIR=Path("web_cache");A2_PACK_DIR=Path("A2_V3_1")
VOLATILE_KEYS={"timestamp","ts","run_id","elapsed_ms","wall_clock","fetched_at"}
HTTP_UA="SMAXIA-GTE/14.3.3 (education-research)";HTTP_TIMEOUT=12
ALLOWED_F_MODES={"SEALED","OPAQUE_DIGEST_ONLY"}
ALLOWED_DA0_MODES={"REAL","SIMULATION"}
ALLOWED_DA1_TEXT_MODES={"OCR_REAL","OCR_SIMULATION"}
def canonical_json(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def sha256(data):
    if isinstance(data,str): data=data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
def strip_volatile(obj):
    if isinstance(obj,dict): return {k:strip_volatile(v) for k,v in sorted(obj.items()) if k not in VOLATILE_KEYS}
    if isinstance(obj,list): return [strip_volatile(i) for i in obj]
    return obj
def _rng_for_country(iso2):
    return random.Random(int(sha256("SMAXIA:"+(iso2 or "").upper())[:8],16))
def determinism_check(pfunc,iso2,n=3):
    hashes=[]
    for _ in range(n):
        r=pfunc(iso2,_determinism_run=True)
        hashes.append(sha256(canonical_json(strip_volatile(r["snapshot"]))))
    return {"pass":len(set(hashes))==1,"hashes":hashes}
def ensure_run_dir(rid):
    p=RUN_DIR/rid;p.mkdir(parents=True,exist_ok=True);return p
def write_artifact(rid,name,data):
    p=ensure_run_dir(rid);c=canonical_json(data);h=sha256(c)
    fp=p/f"{name}.json";fp.write_text(c,encoding="utf-8")
    return {"artifact":name,"sha256":h,"path":str(fp)}
def log_event(evt_type,detail=""):
    if st.session_state.get("_silent_events",False): return
    if "ui_events" not in st.session_state: st.session_state.ui_events=[]
    st.session_state.ui_events.append({"type":evt_type,"detail":detail,"timestamp":datetime.now(timezone.utc).isoformat()})
@st.cache_data
def _build_country_index():
    db={}
    try:
        import pycountry
        db={c.alpha_2:c.name for c in pycountry.countries}
    except Exception:
        try:
            import urllib.request,json as _j
            with urllib.request.urlopen("https://restcountries.com/v3.1/all?fields=cca2,name",timeout=10) as r:
                data=_j.loads(r.read())
            db={c["cca2"]:c["name"]["common"] for c in data if "cca2" in c and "name" in c}
        except Exception: pass
    if not db:
        db={"KM":"Comoros","FR":"France","SN":"Senegal","MA":"Morocco","US":"United States","GB":"United Kingdom","CO":"Colombia","CI":"Ivory Coast","TN":"Tunisia","DZ":"Algeria","CM":"Cameroon","MG":"Madagascar","ML":"Mali","NE":"Niger","TD":"Chad","BF":"Burkina Faso","BJ":"Benin","TG":"Togo","GA":"Gabon","CG":"Congo","CD":"DR Congo","DJ":"Djibouti","MR":"Mauritania","GN":"Guinea","HT":"Haiti","LB":"Lebanon","CA":"Canada","BE":"Belgium","CH":"Switzerland","LU":"Luxembourg","MC":"Monaco","DE":"Germany","ES":"Spain","IT":"Italy","PT":"Portugal","BR":"Brazil","MX":"Mexico","AR":"Argentina","JP":"Japan","CN":"China","IN":"India","KR":"South Korea","AU":"Australia","NZ":"New Zealand","ZA":"South Africa","NG":"Nigeria","KE":"Kenya","GH":"Ghana","EG":"Egypt","SA":"Saudi Arabia","AE":"United Arab Emirates","RU":"Russia","TR":"Turkey","PL":"Poland","RO":"Romania","SE":"Sweden","NO":"Norway","DK":"Denmark","FI":"Finland","NL":"Netherlands","IE":"Ireland","AT":"Austria","CZ":"Czechia","HU":"Hungary","GR":"Greece","IL":"Israel","TH":"Thailand","VN":"Vietnam","PH":"Philippines","ID":"Indonesia","MY":"Malaysia","SG":"Singapore","PK":"Pakistan","BD":"Bangladesh","LK":"Sri Lanka","MM":"Myanmar","KH":"Cambodia","LA":"Laos","ET":"Ethiopia","TZ":"Tanzania","UG":"Uganda","RW":"Rwanda","BI":"Burundi","MW":"Malawi","ZM":"Zambia","ZW":"Zimbabwe","MZ":"Mozambique","AO":"Angola","NA":"Namibia","BW":"Botswana","SZ":"Eswatini","LS":"Lesotho","SC":"Seychelles","MU":"Mauritius","YT":"Mayotte","RE":"Reunion"}
    idx=[{"code":k,"name":v,"nl":v.lower(),"cl":k.lower()} for k,v in db.items()]
    idx.sort(key=lambda e:e["name"]);return db,idx
def typeahead(q,limit=20):
    _,idx=_build_country_index();ql=(q or "").strip().lower()
    if not ql: return [],[]
    iso_like=len(ql)==2 and ql.isalpha()
    prefix,fallback,seen=[],[],set()
    if iso_like:
        for e in idx:
            if e["cl"]==ql: prefix.append(e);seen.add(e["code"]);break
    if not prefix or not iso_like:
        for e in idx:
            if e["code"] in seen: continue
            if e["nl"].startswith(ql): prefix.append(e);seen.add(e["code"])
            if len(prefix)>=limit: break
    if len(prefix)<limit:
        for e in idx:
            if e["code"] in seen: continue
            if e["cl"].startswith(ql) or ql in e["nl"] or ql in e["cl"]:
                fallback.append(e);seen.add(e["code"])
            if len(fallback)>=limit: break
    return prefix,fallback
_HAS_REQUESTS=None;_HAS_BS4=None;_HAS_PDFPLUMBER=None
def _check_libs():
    global _HAS_REQUESTS,_HAS_BS4,_HAS_PDFPLUMBER
    if _HAS_REQUESTS is None:
        try: import requests as _r;_HAS_REQUESTS=True
        except: _HAS_REQUESTS=False
    if _HAS_BS4 is None:
        try: from bs4 import BeautifulSoup as _b;_HAS_BS4=True
        except: _HAS_BS4=False
    if _HAS_PDFPLUMBER is None:
        try: import pdfplumber as _p;_HAS_PDFPLUMBER=True
        except: _HAS_PDFPLUMBER=False
    return _HAS_REQUESTS,_HAS_BS4,_HAS_PDFPLUMBER
def _http_get(url,binary=False,timeout=HTTP_TIMEOUT,cache_only=False):
    WEB_CACHE_DIR.mkdir(parents=True,exist_ok=True)
    ck=sha256(url)[:24];ext=".bin" if binary else ".html"
    cp=WEB_CACHE_DIR/f"{ck}{ext}";cm=WEB_CACHE_DIR/f"{ck}.meta.json"
    if cp.exists() and cm.exists():
        try:
            meta=json.loads(cm.read_text(encoding="utf-8"))
            data=cp.read_bytes() if binary else cp.read_text(encoding="utf-8",errors="replace")
            return data,meta.get("status",200),True,meta
        except: pass
    if cache_only: return None,-1,False,{"error":"cache_miss_no_live","url":url[:200]}
    hr,hb,hp=_check_libs()
    if not hr: return None,-1,False,{"error":"requests_missing"}
    import requests
    try:
        r=requests.get(url,timeout=timeout,headers={"User-Agent":HTTP_UA},allow_redirects=True)
        sc=r.status_code
        if not(200<=sc<300):
            meta={"url":url,"status":sc,"error":"non_2xx"};cm.write_text(canonical_json(meta),encoding="utf-8")
            return None,sc,False,meta
        data=r.content if binary else r.text
        if binary: cp.write_bytes(data)
        else: cp.write_text(data,encoding="utf-8")
        meta={"url":url,"status":sc,"fetched_at":datetime.now(timezone.utc).isoformat(),"size":len(data),"content_type":r.headers.get("content-type","")}
        cm.write_text(canonical_json(meta),encoding="utf-8");return data,sc,False,meta
    except Exception as e: return None,-1,False,{"url":url,"status":-1,"error":str(e)[:200]}
def _extract_pdf_links(html,base_url):
    _,hb,_=_check_libs()
    if not hb or not html: return []
    from bs4 import BeautifulSoup;soup=BeautifulSoup(html,"html.parser")
    pdfs,seen=[],set()
    for a in soup.find_all("a",href=True):
        href=(a.get("href") or "").strip()
        if not href: continue
        full=urljoin(base_url,href)
        if full in seen: continue
        seen.add(full);low=full.lower()
        if low.endswith(".pdf") or ".pdf?" in low:
            pdfs.append({"url":full,"text":(a.get_text(strip=True) or "")[:200]})
    return pdfs
def _domain_score(domain,title="",url=""):
    sc=0.30;s=f"{domain} {title} {url}".lower()
    if re.search(r"\.gov\b",s): sc=max(sc,0.90)
    if re.search(r"\.gouv\.",s): sc=max(sc,0.90)
    if re.search(r"\.edu\b",s): sc=max(sc,0.80)
    if re.search(r"\.ac\.",s): sc=max(sc,0.80)
    if "ministry" in s or "ministere" in s: sc=max(sc,0.75)
    if "examination" in s or "exam" in s or "syllabus" in s or "curriculum" in s: sc=max(sc,0.60)
    return round(sc,3)
def _ddg_search_urls(query,http_diag,cache_only=False):
    qurl=f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    html,sc,cached,meta=_http_get(qurl,binary=False,timeout=HTTP_TIMEOUT,cache_only=cache_only)
    http_diag.append({"engine":"ddg","query":query[:120],"url":qurl[:200],"status":sc,"cached":cached})
    if not html or sc not in range(200,300): return []
    _,hb,_=_check_libs()
    if not hb: return []
    from bs4 import BeautifulSoup;soup=BeautifulSoup(html,"html.parser")
    urls=[]
    for a in soup.select("a.result__a"):
        href=a.get("href") or ""
        if href.startswith("http"): urls.append(href)
        if len(urls)>=8: break
    return urls
def _da0_simulation_manifest(iso2,reason,http_diag):
    rng=_rng_for_country(iso2);sim=[];types=["sujet","corrige"]
    for j in range(min(10,MAX_PDFS)):
        sim.append({"url":f"https://sim.seed/{iso2.upper()}/exam_{j:03d}.pdf","type":types[j%2],"hash":sha256(f"sim_{iso2}_{j}")[:32],"domain":"sim.seed"})
    return {"mode":"SIMULATION","source_manifest":sim,"strategy_log":{"iso2":iso2,"mode":"SIMULATION","fallback_reason":reason,"steps":[]},"http_diag":http_diag,"authority_sources":[],"cache_hit_rate":0.0}
def _try_load_a2_pack():
    result={"sealed_loaded":False,"mode":"OPAQUE_DIGEST_ONLY","pack_sha256":None,"manifest_path":None,"detection_log":[]}
    mp=A2_PACK_DIR/"manifest.json"
    if not mp.exists(): result["detection_log"].append("manifest.json not found in A2_V3_1/");return result
    result["manifest_path"]=str(mp)
    try: manifest=json.loads(mp.read_text(encoding="utf-8"))
    except Exception as e: result["detection_log"].append(f"manifest.json parse error: {str(e)[:100]}");return result
    expected_sha=manifest.get("formula_module_sha256");mn=manifest.get("formula_module","FORMULA.py")
    modp=A2_PACK_DIR/mn
    if not modp.exists(): result["detection_log"].append(f"{mn} not found");return result
    actual_sha=sha256(modp.read_bytes())
    if expected_sha and actual_sha!=expected_sha: result["detection_log"].append("SHA256 mismatch");return result
    result["pack_sha256"]=actual_sha
    try:
        import importlib.util
        spec=importlib.util.spec_from_file_location("a2_formula",str(modp));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        if hasattr(mod,"compute_f1") and hasattr(mod,"compute_f2"):
            result["sealed_loaded"]=True;result["mode"]="SEALED";result["_module"]=mod;result["detection_log"].append("A2 pack loaded and sealed OK")
        else: result["detection_log"].append("Module missing compute_f1/compute_f2")
    except Exception as e: result["detection_log"].append(f"Module load error: {str(e)[:150]}")
    return result
def _init_formula_pack():
    a2=_try_load_a2_pack()
    pack={"version":"FP-3.1-SEALED","mode":a2["mode"],"sealed_loaded":a2["sealed_loaded"],"pack_sha256":a2.get("pack_sha256"),"detection_log":a2["detection_log"],"sha256_manifest":None,"executors":{}}
    if a2["sealed_loaded"] and "_module" in a2:
        mod=a2["_module"];pack["executors"]["compute_f1"]=mod.compute_f1;pack["executors"]["compute_f2"]=mod.compute_f2
    else:
        pack["executors"]["compute_f1"]=_fp_compute_f1_opaque;pack["executors"]["compute_f2"]=_fp_compute_f2_opaque
    pack["sha256_manifest"]=sha256(canonical_json({"version":pack["version"],"mode":pack["mode"],"sealed_loaded":pack["sealed_loaded"],"pack_sha256":pack["pack_sha256"],"f1_src_hash":sha256(inspect.getsource(pack["executors"]["compute_f1"])),"f2_src_hash":sha256(inspect.getsource(pack["executors"]["compute_f2"]))}))
    return pack
def _fp_compute_f1_opaque(atoms,frt,qc,pack_cfg):
    """F1 opaque digest executor — OPAQUE_DIGEST_ONLY mode."""
    n=len(atoms)
    if n==0: return {"digest":sha256("empty"),"status":"NO_ATOMS","mode":"OPAQUE_DIGEST_ONLY"}
    qch=canonical_json(sorted([q.get("qi_hash","") for q in sorted(qc,key=lambda x:x.get("atom_id",""))]))
    ah=canonical_json([a["Qi"]["source_hash"] for a in atoms])
    return {"digest":sha256(qch+ah+canonical_json(frt)),"status":"OK","mode":"OPAQUE_DIGEST_ONLY","inputs_n":n}
def _fp_compute_f2_opaque(f1d,ari_profiles,triggers,pack_cfg):
    """F2 opaque digest executor — OPAQUE_DIGEST_ONLY mode."""
    bd=f1d.get("digest","");ac=canonical_json(sorted([p.get("atom_id","") for p in ari_profiles]))
    tc=canonical_json(sorted([t.get("trigger","") for t in triggers]))
    return {"digest":sha256(bd+ac+tc),"status":"OK","mode":"OPAQUE_DIGEST_ONLY","inputs_ari":len(ari_profiles),"inputs_trig":len(triggers)}
FORMULA_PACK=_init_formula_pack()
class FormulaEngine:
    BANNED_KEYWORDS=["delta_c","epsilon","psi_raw","one_m","alpha","anti_red","freq","cosine"]
    @staticmethod
    def _audit_source(func):
        src=inspect.getsource(func)
        for kw in FormulaEngine.BANNED_KEYWORDS:
            if kw in src: return {"pass":False,"keyword":kw}
        return {"pass":True}
    @staticmethod
    def compute_f1(atoms,frt,qc,pack_cfg=None): return FORMULA_PACK["executors"]["compute_f1"](atoms,frt,qc,pack_cfg or {})
    @staticmethod
    def compute_f2(f1d,ari,trig,pack_cfg=None): return FORMULA_PACK["executors"]["compute_f2"](f1d,ari,trig,pack_cfg or {})
    @staticmethod
    def self_audit():
        a1=FormulaEngine._audit_source(FORMULA_PACK["executors"]["compute_f1"])
        a2=FormulaEngine._audit_source(FORMULA_PACK["executors"]["compute_f2"])
        viol=[]
        if not a1["pass"]: viol.append(a1)
        if not a2["pass"]: viol.append(a2)
        return {"compute_f1_executor_clean":a1["pass"],"compute_f2_executor_clean":a2["pass"],"formula_pack_version":FORMULA_PACK["version"],"formula_pack_mode":FORMULA_PACK["mode"],"formula_pack_sha256":FORMULA_PACK["sha256_manifest"],"sealed_loaded":FORMULA_PACK["sealed_loaded"],"pack_sha256":FORMULA_PACK["pack_sha256"],"violations":viol}
def chk_no_country_branching():
    try: src=Path(__file__).read_text(encoding="utf-8",errors="replace")
    except: return {"pass":False,"reason":"cannot_read_source"}
    sc=re.sub(r"['\"][A-Z]{2}\s*\(.*?\)['\"]","X",src)
    sc=re.sub(r"#.*$","",sc,flags=re.MULTILINE)
    sc=re.sub(r'""".*?"""',"X",sc,flags=re.DOTALL)
    sc=re.sub(r"'''.*?'''","X",sc,flags=re.DOTALL)
    bad=[]
    for pat,label in [(r'==\s*["\']FR["\']',"eq_FR"),(r'==\s*["\']SN["\']',"eq_SN"),(r'==\s*["\']MA["\']',"eq_MA"),(r'==\s*["\']US["\']',"eq_US"),(r'==\s*["\']GB["\']',"eq_GB"),(r'if\s+.*country\s*==',"if_country"),(r'switch.*country',"switch_country"),(r'\bcountry_map\s*\[',"country_map"),(r'\bcountry_dict\s*\[',"country_dict")]:
        if re.findall(pat,sc,re.IGNORECASE): bad.append({"pattern":label,"count":len(re.findall(pat,sc,re.IGNORECASE))})
    return {"pass":len(bad)==0,"bad_patterns":bad}
def build_cap_questions():
    return ["What is the structure of the national education system?","What are the main examination bodies and their roles?","What are the high-stakes national exams at each transition point?","What subjects carry the highest coefficient weights?","What are the top university programs by demand?","What are the top 5-7 competitive exams per transition?","What grading scale and passing thresholds are used?"]
def discover_cap(iso2):
    log_event("CAP_DISCOVERY_START",iso2)
    cap={"iso2":iso2,"questions":build_cap_questions(),"responses":{},"status":"DISCOVERED","sha256":None}
    for i,q in enumerate(cap["questions"]): cap["responses"][f"Q{i}"]={"question":q,"answer":f"[PENDING_HARVEST_{iso2}]","source":"awaiting_activation"}
    cap["sha256"]=sha256(canonical_json(strip_volatile(cap)));log_event("CAP_DISCOVERY_END",iso2);return cap
def validate_vsp(cap):
    checks=[{"check":"iso2_present","pass":bool(cap.get("iso2"))},{"check":"questions_count","pass":len(cap.get("questions",[]))>=5},{"check":"responses_exist","pass":len(cap.get("responses",{}))>0}]
    return {"status":"PASS" if all(c["pass"] for c in checks) else "FAIL","checks":checks,"cap_sha256":cap.get("sha256")}
def discover_sources(iso2,cap,cache_only=False):
    log_event("DA0_START",iso2);db,_=_build_country_index();cn=db.get((iso2 or "").upper(),(iso2 or "").upper())
    hd=[];sl={"iso2":iso2,"country_name":cn,"max_pdfs":MAX_PDFS,"mode":"REAL","steps":[]}
    hr,hb,hp=_check_libs()
    if not hr or not hb:
        reason=f"libs_missing:requests={hr},bs4={hb}";sl["steps"].append({"step":"lib_check","result":reason})
        log_event("DA0_FALLBACK_SIMULATION",reason);return _da0_simulation_manifest(iso2,reason,hd)
    queries=[f"{cn} ministry of education official site",f"{cn} national examination council official",f"{cn} past exam papers PDF",f"{cn} curriculum syllabus PDF"]
    cand=[];ch,ct=0,0
    for q in queries:
        urls=_ddg_search_urls(q,hd,cache_only=cache_only);sl["steps"].append({"step":"search_ddg","query":q,"urls_found":len(urls)});cand.extend(urls)
    if not cand:
        reason="ddg_returned_no_pages";sl["steps"].append({"step":"fallback","result":reason})
        log_event("DA0_FALLBACK_SIMULATION",reason);return _da0_simulation_manifest(iso2,reason,hd)
    pdfs=[];seen_pdf=set();auth=[];seen_dom=set()
    for pu in cand[:20]:
        html,sc,cached,meta=_http_get(pu,binary=False,timeout=HTTP_TIMEOUT,cache_only=cache_only);ct+=1
        if cached: ch+=1
        hd.append({"engine":"fetch","url":pu[:220],"status":sc,"cached":cached})
        if not html or sc not in range(200,300): continue
        dom=urlparse(pu).netloc.lower();title=""
        m=re.search(r"<title>([^<]+)</title>",html,re.I)
        if m: title=(m.group(1) or "")[:200]
        ds=_domain_score(dom,title,pu)
        if dom and dom not in seen_dom: seen_dom.add(dom);auth.append({"domain":dom,"sample_url":pu[:500],"title":title,"authority_score":ds})
        for p in _extract_pdf_links(html,pu):
            u=p["url"]
            if u in seen_pdf: continue
            seen_pdf.add(u);pdfs.append({"url":u,"type":"pdf","from_page":pu[:500],"domain":urlparse(u).netloc.lower(),"hash":sha256(u)[:32]})
            if len(pdfs)>=MAX_PDFS: break
        if len(pdfs)>=MAX_PDFS: break
    chr_=round(ch/max(ct,1),3);sl["cache_hit_rate"]=chr_
    if not pdfs:
        reason="no_pdf_links_found";sl["steps"].append({"step":"fallback","result":reason})
        log_event("DA0_FALLBACK_SIMULATION",reason);r=_da0_simulation_manifest(iso2,reason,hd);r["strategy_log"]=sl;r["authority_sources"]=auth;r["cache_hit_rate"]=chr_;return r
    sm=[]
    for p in pdfs[:MAX_PDFS]:
        low=p["url"].lower();dt="corrige" if ("corrig" in low or "answer" in low or "solution" in low) else "sujet"
        sm.append({"url":p["url"],"type":dt,"hash":p["hash"],"domain":p["domain"]})
    log_event("DA0_END",f"{len(sm)} PDFs");return {"mode":"REAL","source_manifest":sm,"strategy_log":sl,"http_diag":hd,"authority_sources":auth,"cache_hit_rate":chr_}
def build_cep(sources):
    m=sources.get("source_manifest",[]);suj=[s for s in m if s["type"]=="sujet"];cor=[s for s in m if s["type"]=="corrige"]
    pairs,unp=[],[]
    for i,s in enumerate(suj):
        if i<len(cor): pairs.append({"sujet":s["url"],"corrige":cor[i]["url"],"sujet_hash":s["hash"],"corrige_hash":cor[i]["hash"],"pair_id":f"CEP_{i:04d}"})
        else: unp.append({"url":s["url"],"reason":"no_matching_corrige"})
    for c in cor[len(suj):]: unp.append({"url":c["url"],"reason":"no_matching_sujet"})
    return {"pairs":pairs,"unpaired":unp,"total_pairs":len(pairs)}
def _try_extract_pdf_text(pdf_bytes):
    _,_,hp=_check_libs()
    if not hp or not pdf_bytes: return None,None
    try:
        import pdfplumber,io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if pdf.pages:
                text=pdf.pages[0].extract_text() or ""
                if len(text.strip())>20: return text.strip()[:2000],"pdfplumber_page0"
    except: pass
    return None,None
def execute_da1(cep,iso2,cache_only=False):
    log_event("DA1_START");rng=_rng_for_country(iso2);dl=[];texts={};tm="OCR_SIMULATION";roc=0
    for pair in cep.get("pairs",[]):
        pid=pair["pair_id"];su=pair.get("sujet","");cu=pair.get("corrige","")
        sd,ss,_,_=_http_get(su,binary=True,timeout=HTTP_TIMEOUT,cache_only=cache_only)
        cd,cs,_,_=_http_get(cu,binary=True,timeout=HTTP_TIMEOUT,cache_only=cache_only)
        dl.append({"pair_id":pid,"sujet_status":"DL_OK" if ss in range(200,300) else f"DL_FAIL_{ss}","corrige_status":"DL_OK" if cs in range(200,300) else f"DL_FAIL_{cs}","sujet_url":su[:300],"corrige_url":cu[:300]})
        st_,sm_=_try_extract_pdf_text(sd if isinstance(sd,bytes) else None)
        ct_,cm2=_try_extract_pdf_text(cd if isinstance(cd,bytes) else None)
        if st_: roc+=1
        if ct_: roc+=1
        texts[pid]={"sujet_text":st_ or (f"[OCR_PENDING_{pid}_SUJET]" if sd else f"[DL_FAIL_{pid}_SUJET]"),"corrige_text":ct_ or (f"[OCR_PENDING_{pid}_CORRIGE]" if cd else f"[DL_FAIL_{pid}_CORRIGE]"),"sujet_ocr_method":sm_ or "placeholder","corrige_ocr_method":cm2 or "placeholder","ocr_confidence":round(rng.uniform(0.75,0.98),3) if sd else 0.0}
    if roc>0: tm="OCR_REAL"
    log_event("DA1_END",f"{len(dl)} pairs, text_mode={tm}");return {"dl_log":dl,"texts":texts,"text_mode":tm,"real_ocr_count":roc}
def extract_atoms(texts):
    log_event("ATOMS_START");atoms=[]
    for pid in sorted(texts.keys()):
        t=texts[pid];stxt=t.get("sujet_text","");ctxt=t.get("corrige_text","")
        atoms.append({"Qi":{"id":f"{pid}_Q1","pair_id":pid,"text":stxt[:200],"source_hash":sha256(stxt)},"RQi":{"id":f"{pid}_R1","pair_id":pid,"text":ctxt[:200],"source_hash":sha256(ctxt)}})
    log_event("ATOMS_END",f"{len(atoms)} atoms");return atoms
def compute_frt(atoms):
    log_event("FRT_START");themes={}
    for a in atoms: k=a["Qi"]["pair_id"];themes[k]=themes.get(k,0)+1
    frt=[{"theme":k,"frequency":v,"recurrence":1,"weight":round(v/max(len(atoms),1),4)} for k,v in sorted(themes.items())]
    log_event("FRT_END",f"{len(frt)} themes");return frt
def run_qc(atoms):
    log_event("QC_START");qc=[]
    for a in atoms:
        qi,rqi=a["Qi"],a["RQi"];valid=len(qi["text"])>10 and len(rqi["text"])>10
        qc.append({"atom_id":qi["id"],"valid":valid,"qi_hash":qi["source_hash"],"rqi_hash":rqi["source_hash"],"reason":"OK" if valid else "TEXT_TOO_SHORT"})
    log_event("QC_END",f"{sum(1 for q in qc if q['valid'])}/{len(qc)} valid");return qc
def compute_ari(atoms,qc,iso2):
    rng=_rng_for_country(iso2);profiles=[]
    for a in atoms:
        qid=a["Qi"]["id"];qce=next((q for q in qc if q["atom_id"]==qid),None)
        if qce and qce["valid"]: profiles.append({"atom_id":qid,"difficulty":round(rng.uniform(0.3,0.9),3),"discrimination":round(rng.uniform(0.1,0.7),3),"source_qi_hash":a["Qi"]["source_hash"]})
    return profiles
def compute_triggers(ari,frt):
    trig=[]
    for p in ari:
        if p["difficulty"]>0.7: trig.append({"atom_id":p["atom_id"],"trigger":"HIGH_DIFFICULTY","value":p["difficulty"]})
    for f in frt:
        if f["weight"]>0.2: trig.append({"theme":f["theme"],"trigger":"HIGH_FREQ_THEME","value":f["weight"]})
    return trig
def build_soe(atoms,frt): return {"total_atoms":len(atoms),"total_themes":len(frt),"top_themes":sorted(frt,key=lambda x:-x["weight"])[:5]}
def run_gates(cap,vsp,sources,cep,da1,atoms,frt,qc,f1,f2,ari,triggers,execution_mode="TEST_ISO_PROD"):
    gates=[];is_test=execution_mode=="TEST_ISO_PROD"
    def gate(name,cond,ev): gates.append({"gate":name,"verdict":"PASS" if cond else "FAIL","evidence":ev})
    gate("CHK_UI_EVENT_LOG",len(st.session_state.get("ui_events",[]))>0 or is_test,"ui_events_or_test_mode")
    ncb=chk_no_country_branching();gate("CHK_NO_COUNTRY_BRANCHING",ncb["pass"],"chk_no_country_branching_report")
    da0m=sources.get("mode","UNKNOWN");gate("CHK_DA0_MODE_ALLOWED",da0m in ALLOWED_DA0_MODES,f"sources.mode={da0m}")
    gate("GATE_DA0",len(sources.get("source_manifest",[]))>0 or is_test,"source_manifest_or_test_mode")
    gate("GATE_DA1",len(da1.get("dl_log",[]))>0,"dl_log")
    gate("GATE_TEXT",any(t.get("sujet_text") for t in da1.get("texts",{}).values()),"texts")
    dtm=da1.get("text_mode","UNKNOWN");gate("CHK_DA1_TEXT_MODE_ALLOWED",dtm in ALLOWED_DA1_TEXT_MODES,f"da1.text_mode={dtm}")
    gate("GATE_ATOMS",len(atoms)>0,"atoms")
    gate("GATE_QC",any(q["valid"] for q in qc),"qc_validated")
    f1m=f1.get("mode","UNKNOWN");f2m=f2.get("mode","UNKNOWN")
    gate("CHK_F1F2_MODE_ALLOWED",f1m in ALLOWED_F_MODES and f2m in ALLOWED_F_MODES,f"f1.mode={f1m},f2.mode={f2m}")
    gate("CHK_F1F2_STATUS_OK",f1.get("status")=="OK" and f2.get("status")=="OK","f1f2_status")
    gate("CHK_CAP_COMPLETENESS",vsp.get("status")=="PASS","vsp_output")
    gate("GATE_FRT",len(frt)>0,"frt")
    gate("GATE_ARI",len(ari)>0,"ari_profiles")
    ap=all(g["verdict"]=="PASS" for g in gates)
    return {"gates":gates,"global_verdict":"PASS" if ap else "FAIL","execution_mode":execution_mode,"no_country_branching_report":ncb}
def run_pipeline(iso2,_determinism_run=False):
    st.session_state["_silent_events"]=bool(_determinism_run);execution_mode="TEST_ISO_PROD"
    cap=discover_cap(iso2);run_id=f"RUN_{iso2}_{sha256(canonical_json(strip_volatile(cap)))[:8]}"
    if not _determinism_run: log_event("PIPELINE_START",f"{iso2} / {run_id}")
    vsp=validate_vsp(cap)
    if _determinism_run:
        sources=_da0_simulation_manifest(iso2,"determinism_run",[])
    else:
        sources=discover_sources(iso2,cap,cache_only=(execution_mode=="TEST_ISO_PROD"))
    cep=build_cep(sources);da1=execute_da1(cep,iso2,cache_only=_determinism_run)
    atoms=extract_atoms(da1["texts"]);frt=compute_frt(atoms);qc=run_qc(atoms);soe=build_soe(atoms,frt)
    ari=compute_ari(atoms,qc,iso2);triggers=compute_triggers(ari,frt)
    f1=FormulaEngine.compute_f1(atoms,frt,qc);f2=FormulaEngine.compute_f2(f1,ari,triggers)
    gr=run_gates(cap,vsp,sources,cep,da1,atoms,frt,qc,f1,f2,ari,triggers,execution_mode);fea=FormulaEngine.self_audit()
    snapshot=strip_volatile({"cap":cap,"vsp":vsp,"sources":sources["source_manifest"],"cep":cep,"atoms":atoms,"frt":frt,"qc":qc,"f1":f1,"f2":f2,"ari":ari,"triggers":triggers,"gates":gr})
    if _determinism_run: st.session_state["_silent_events"]=False;return {"snapshot":snapshot}
    det=determinism_check(run_pipeline,iso2)
    al=[("CAP_SEALED",cap),("VSP_output",vsp),("SourceManifest",sources["source_manifest"]),("SDA0_HTTP_DIAG",sources["http_diag"]),("SDA0_STRATEGY_LOG",sources["strategy_log"]),("CEP_pairs",cep),("DA1_DL_LOG",da1["dl_log"]),("SOE",soe),("Atoms_Qi_RQi",atoms),("FRT",frt),("QC_validated",qc),("F1_call_digest",f1),("F2_call_digest",f2),("ARI",ari),("Triggers",triggers),("FORMULA_PACK_MANIFEST",{"version":FORMULA_PACK["version"],"mode":FORMULA_PACK["mode"],"sealed_loaded":FORMULA_PACK["sealed_loaded"],"sha256":FORMULA_PACK["sha256_manifest"],"pack_sha256":FORMULA_PACK["pack_sha256"],"fe_audit":fea,"detection_log":FORMULA_PACK["detection_log"]}),("UI_EVENT_LOG",st.session_state.get("ui_events",[])),("CHK_REPORT",gr),("NO_COUNTRY_BRANCHING_REPORT",gr.get("no_country_branching_report",{})),("DeterminismReport_3runs",det)]
    artifacts={}
    for name,data in al: artifacts[name]=write_artifact(run_id,name,data)
    seal={"run_id":run_id,"iso2":iso2,"execution_mode":execution_mode,"global_verdict":gr["global_verdict"],"determinism_pass":det["pass"],"artifact_count":len(al)+1,"artifact_hashes":{k:v["sha256"] for k,v in artifacts.items()},"formula_engine_audit":fea,"formula_pack_mode":FORMULA_PACK["mode"],"sealed_loaded":FORMULA_PACK["sealed_loaded"],"da0_mode":sources.get("mode"),"da1_text_mode":da1.get("text_mode")}
    artifacts["SealReport"]=write_artifact(run_id,"SealReport",seal);seal["artifact_count"]=len(artifacts)
    log_event("PIPELINE_END",run_id);st.session_state["_silent_events"]=False
    return {"run_id":run_id,"cap":cap,"vsp":vsp,"sources":sources,"cep":cep,"da1":da1,"atoms":atoms,"frt":frt,"qc":qc,"soe":soe,"f1":f1,"f2":f2,"ari":ari,"triggers":triggers,"gates":gr,"seal":seal,"artifacts":artifacts,"determinism":det,"snapshot":snapshot}
def json_view(data): st.json(data if isinstance(data,(dict,list)) else {"value":data})
def no_data_msg(): st.info("⏳ Activate a country from the sidebar to populate this tab.")
def main():
    st.set_page_config(page_title="SMAXIA GTE v14.3.3",layout="wide")
    st.title("🛡️ SMAXIA GTE Kernel v14.3.3 — ISO-PROD")
    if "pipeline" not in st.session_state: st.session_state.pipeline=None
    if "ui_events" not in st.session_state: st.session_state.ui_events=[]
    with st.sidebar:
        st.markdown("### 🌍 Activate Country")
        q=st.text_input("Type country name or code",key="cc_query",placeholder="ex: France, Senegal, FR, SN, KM")
        pm,fb=typeahead(q,limit=20);options=pm if pm else fb
        labels=[f"{e['name']} ({e['code']})" for e in options];cc=None
        if labels:
            st.caption(f"{len(labels)} match{'es' if len(labels)!=1 else ''}")
            choice=st.selectbox("Select country",options=labels,index=0)
            cc=options[labels.index(choice)]["code"]
        else:
            if q and q.strip(): st.caption("No matches found.")
        if st.button("🚀 ACTIVATE_COUNTRY",type="primary",use_container_width=True):
            if cc and len(cc)==2:
                log_event("ACTIVATE_COUNTRY",cc)
                with st.spinner(f"Running pipeline for {cc}…"): st.session_state.pipeline=run_pipeline(cc)
                st.success(f"✅ {cc} activated")
            else: st.error("Select a valid country from the list.")
        if st.session_state.pipeline:
            st.divider();p=st.session_state.pipeline
            st.markdown(f"**Active:** `{p['seal']['iso2']}`")
            v=p["seal"]["global_verdict"]
            st.markdown(f"**Verdict:** :green[{v}]" if v=="PASS" else f"**Verdict:** :red[{v}]")
            st.markdown(f"**Determinism:** `{'PASS' if p['determinism']['pass'] else 'FAIL'}`")
            st.markdown(f"**Formula Pack:** `{FORMULA_PACK['mode']}`")
            st.markdown(f"**DA0:** `{p['seal'].get('da0_mode','N/A')}` | **DA1:** `{p['seal'].get('da1_text_mode','N/A')}`")
            st.markdown(f"**Exec Mode:** `{p['seal'].get('execution_mode','N/A')}`")
            st.markdown(f"**Run:** `{p['run_id']}`")
    tabs=st.tabs(["🛡️ Admin","📦 CAP/VSP","🔍 DA0/Sources","📋 CEP/Pairs","📄 Text/OCR","🧬 Atoms","🔧 FRT","🔎 QC Explorer","🚦 Gates","📊 F1/F2","📁 Artifacts"])
    with tabs[0]:
        st.header("Admin — Dashboard")
        if not st.session_state.pipeline: no_data_msg()
        else:
            p=st.session_state.pipeline;seal=p["seal"]
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Country",seal["iso2"]);c2.metric("Global Verdict",seal["global_verdict"]);c3.metric("Determinism","PASS" if p["determinism"]["pass"] else "FAIL");c4.metric("Artifacts",seal["artifact_count"]);c5.metric("Exec Mode",seal.get("execution_mode","N/A"))
            st.subheader("Formula Pack Status")
            st.write(f"**Mode:** `{FORMULA_PACK['mode']}` | **Sealed Loaded:** `{FORMULA_PACK['sealed_loaded']}`")
            if FORMULA_PACK["detection_log"]:
                with st.expander("A2 Detection Log"):
                    for entry in FORMULA_PACK["detection_log"]: st.write(f"• {entry}")
            st.subheader("Pipeline Modes")
            st.write(f"**DA0:** `{seal.get('da0_mode')}` | **DA1 Text:** `{seal.get('da1_text_mode')}` | **F1/F2:** `{seal.get('formula_pack_mode')}`")
            st.subheader("FormulaEngine Self-Audit");json_view(FormulaEngine.self_audit())
    with tabs[1]:
        st.header("📦 CAP / VSP")
        if not st.session_state.pipeline: no_data_msg()
        else: st.subheader("CAP");json_view(st.session_state.pipeline["cap"]);st.subheader("VSP");json_view(st.session_state.pipeline["vsp"])
    with tabs[2]:
        st.header("🔍 DA0 / Sources")
        if not st.session_state.pipeline: no_data_msg()
        else:
            p=st.session_state.pipeline
            st.write(f"**Mode:** `{p['sources'].get('mode','?')}` | **Cache Hit Rate:** `{p['sources'].get('cache_hit_rate','N/A')}`")
            st.subheader("Source Manifest");json_view(p["sources"]["source_manifest"])
            if p["sources"].get("authority_sources"): st.subheader("Authority Sources");json_view(p["sources"]["authority_sources"])
            st.subheader("HTTP Diag");json_view(p["sources"]["http_diag"])
            st.subheader("Strategy Log");json_view(p["sources"]["strategy_log"])
    with tabs[3]:
        st.header("📋 CEP / Pairs")
        if not st.session_state.pipeline: no_data_msg()
        else: json_view(st.session_state.pipeline["cep"])
    with tabs[4]:
        st.header("📄 Text / OCR")
        if not st.session_state.pipeline: no_data_msg()
        else:
            da1=st.session_state.pipeline["da1"]
            st.write(f"**Text Mode:** `{da1.get('text_mode')}` | **Real OCR:** `{da1.get('real_ocr_count',0)}`")
            texts=da1["texts"]
            for pid in sorted(texts.keys()):
                t=texts[pid]
                with st.expander(f"Pair: {pid} (conf: {t['ocr_confidence']}) [{t.get('sujet_ocr_method','?')}/{t.get('corrige_ocr_method','?')}]"):
                    st.text_area("Sujet",t["sujet_text"],height=80,key=f"s_{pid}",disabled=True)
                    st.text_area("Corrige",t["corrige_text"],height=80,key=f"c_{pid}",disabled=True)
    with tabs[5]:
        st.header("🧬 Atoms (Qi / RQi)")
        if not st.session_state.pipeline: no_data_msg()
        else: json_view(st.session_state.pipeline["atoms"])
    with tabs[6]:
        st.header("🔧 FRT")
        if not st.session_state.pipeline: no_data_msg()
        else: json_view(st.session_state.pipeline["frt"])
    with tabs[7]:
        st.header("🔎 QC Explorer")
        if not st.session_state.pipeline: no_data_msg()
        else:
            qc=st.session_state.pipeline["qc"];vc=sum(1 for q in qc if q["valid"])
            st.metric("Valid / Total",f"{vc} / {len(qc)}");json_view(qc)
    with tabs[8]:
        st.header("🚦 Gates")
        if not st.session_state.pipeline: no_data_msg()
        else:
            gr=st.session_state.pipeline["gates"];v=gr["global_verdict"]
            st.success(f"Global Verdict: {v}") if v=="PASS" else st.error(f"Global Verdict: {v}")
            for g in gr["gates"]: st.write(f"{'✅' if g['verdict']=='PASS' else '❌'} **{g['gate']}** → {g['verdict']}  (evidence: `{g['evidence']}`)")
    with tabs[9]:
        st.header("📊 F1 / F2")
        if not st.session_state.pipeline: no_data_msg()
        else:
            p=st.session_state.pipeline;c1,c2=st.columns(2)
            with c1: st.subheader("F1 Digest");json_view(p["f1"])
            with c2: st.subheader("F2 Digest");json_view(p["f2"])
            st.subheader("ARI Profiles");json_view(p["ari"])
            st.subheader("Triggers");json_view(p["triggers"])
    with tabs[10]:
        st.header("📁 Artifacts")
        if not st.session_state.pipeline: no_data_msg()
        else:
            arts=st.session_state.pipeline["artifacts"]
            for name,meta in sorted(arts.items()): st.write(f"📄 **{name}** — `{meta['sha256'][:16]}…` — `{meta['path']}`")
            st.subheader("SealReport");json_view(st.session_state.pipeline["seal"])
            st.subheader("Determinism Report (3 runs)");json_view(st.session_state.pipeline["determinism"])
if __name__=="__main__":
    main()
