#!/usr/bin/env python3
"""SMAXIA GTE CORE v15.1.0 — ISO-PROD
Pipeline 11 Blocs: ACTIVATE→DA0→CAP→DA1→ATOMS→IA1_MINER→CLUSTERING→IA1_BUILDER→IA2_JUDGE→F1F2→COVERAGE+SEAL
UI: 6 onglets Mission Control (CAP, SUJETS, Qi/RQi, QC, KPI, MAPPING)
Fixes: st.secrets for API keys, Wikidata P37 for lang detection, Brave L2 snapshot."""
import streamlit as st
import json,hashlib,re,os,importlib.util,io
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote_plus,urljoin,urlparse
RUN_DIR,CACHE_DIR=Path("run"),Path("web_cache")
VOLATILE={"timestamp","ts","run_id","elapsed_ms","wall_clock","fetched_at"}
EXEC_MODE="TEST";VERSION="15.1.0"
def cj(o):return json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def sha(d):return hashlib.sha256((d if isinstance(d,bytes)else d.encode("utf-8"))).hexdigest()
def sv(o):
    if isinstance(o,dict):return{k:sv(v)for k,v in sorted(o.items())if k not in VOLATILE}
    if isinstance(o,list):return[sv(i)for i in o]
    return o
def w_art(rid,name,data):
    p=RUN_DIR/rid;p.mkdir(parents=True,exist_ok=True)
    c=cj(data);fp=p/f"{name}.json";fp.write_text(c,encoding="utf-8");return{"artifact":name,"sha256":sha(c),"path":str(fp)}
def log_ev(t,d=""):
    if st.session_state.get("_sil"):return
    st.session_state.setdefault("ui_ev",[]).append({"type":t,"detail":d,"ts":datetime.now(timezone.utc).isoformat()})
def _secret(key):
    """st.secrets FIRST (Streamlit Cloud canonical), then os.environ fallback."""
    try:return st.secrets[key]
    except:return os.environ.get(key)
@st.cache_data
def _cdb():
    db={}
    try:
        import pycountry;db={c.alpha_2:c.name for c in pycountry.countries}
    except:
        try:
            import urllib.request
            with urllib.request.urlopen("https://restcountries.com/v3.1/all?fields=cca2,name",timeout=10)as r:
                db={c["cca2"]:c["name"]["common"]for c in json.loads(r.read())if"cca2"in c}
        except:pass
    return db,sorted([{"c":k,"n":v,"nl":v.lower(),"cl":k.lower()}for k,v in db.items()],key=lambda e:e["n"])
def typeahead(q,lim=20):
    _,idx=_cdb();ql=(q or"").strip().lower()
    if not ql:return[]
    hits,seen=[],set()
    if len(ql)==2 and ql.isalpha():
        for e in idx:
            if e["cl"]==ql:hits.append(e);seen.add(e["c"]);break
    for e in idx:
        if e["c"]not in seen and e["nl"].startswith(ql):hits.append(e);seen.add(e["c"])
        if len(hits)>=lim:break
    if len(hits)<lim:
        for e in idx:
            if e["c"]not in seen and(ql in e["nl"]or ql in e["cl"]):hits.append(e);seen.add(e["c"])
            if len(hits)>=lim:break
    return hits
def http_get(url,binary=False,timeout=15,cache_only=False):
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    ck=sha(url)[:24];ext=".bin"if binary else".html"
    cp,cm=CACHE_DIR/f"{ck}{ext}",CACHE_DIR/f"{ck}.meta.json"
    if cp.exists()and cm.exists():
        try:
            meta=json.loads(cm.read_text(encoding="utf-8"))
            data=cp.read_bytes()if binary else cp.read_text(encoding="utf-8",errors="replace")
            return data,meta.get("status",200),True,meta
        except:pass
    if cache_only:return None,-1,False,{"error":"cache_miss"}
    try:
        import requests as rq
        r=rq.get(url,timeout=timeout,headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},allow_redirects=True)
        sc=r.status_code
        if 200<=sc<300:
            data=r.content if binary else r.text
            if binary:cp.write_bytes(data)
            else:cp.write_text(data,encoding="utf-8")
            meta={"url":url,"status":sc,"fetched_at":datetime.now(timezone.utc).isoformat(),"size":len(data)}
            cm.write_text(cj(meta),encoding="utf-8");return data,sc,False,meta
        return None,sc,False,{"url":url,"status":sc}
    except Exception as e:return None,-1,False,{"url":url,"error":str(e)[:200]}
def _links(html,base,fn=None):
    try:
        raw=re.findall(r'href=["\']([^"\']+)',html or"");out=[]
        for r2 in raw:
            u=urljoin(base,r2);txt=""
            pos=html.find(r2)
            if pos>=0:
                m=re.search(r'>([^<]{1,120})</a',html[pos:pos+500],re.I)
                if m:txt=m.group(1).strip()
            if fn and not fn(u,txt):continue
            out.append({"url":u,"text":txt})
        return out
    except:return[]
# ═══ BLOC 1: ACTIVATE_COUNTRY ═══
def bloc1_activate(iso2):
    log_ev("BLOC1_ACTIVATE",iso2);db,_=_cdb();cn=db.get(iso2)
    if not cn:return{"status":"FAIL","gate":"GATE_ISO2_VALID","iso2":iso2}
    return{"status":"OK","iso2":iso2,"country_name":cn,"languages":[],"gate":"GATE_ISO2_VALID"}
# ═══ BLOC 2: DA0 Authority Discovery ═══
def _dom_score(dom):
    for kw,v in[(".gov",0.9),(".gouv",0.9),(".nic.",0.85),(".edu",0.8),(".ac.",0.7)]:
        if kw in dom.lower():return v
    return 0.3
def _is_sub(dom,roots):
    for r in roots:
        if dom==r or dom.endswith("."+r):return True
    return False
def _auth_score(url,html):
    dom=urlparse(url).netloc.lower();reasons=[];sc=0.3
    for kw,v,tag in[(".gov",0.9,"TLD_GOV"),(".gouv",0.9,"TLD_GOUV"),(".edu",0.8,"TLD_EDU"),(".ac.",0.7,"TLD_AC"),(".nic.",0.85,"TLD_NIC")]:
        if kw in dom:sc=max(sc,v);reasons.append(tag)
    low=(html or"").lower()[:8000]
    for kw,tag in[("ministry","OFFICIAL_EDU"),("ministère","OFFICIAL_EDU"),("education","OFFICIAL_EDU"),("examination","OFFICIAL_EXAM"),("question paper","EXAM_CONTENT"),("past paper","EXAM_CONTENT"),("annales","EXAM_CONTENT"),("sujet","EXAM_CONTENT")]:
        if kw in low:reasons.append(tag)
    if any(k in reasons for k in["OFFICIAL_EDU","OFFICIAL_EXAM"]):sc=max(sc,0.75)
    if"EXAM_CONTENT"in reasons:sc=max(sc,0.7)
    return{"score":round(sc,2),"reasons":reasons[:10]}
def _brave_search(queries,diag,max_per=10):
    bk=_secret("BRAVE_KEY")
    if not bk:return[],[]
    cands=[];snaps=[]
    try:
        import requests as rq
        for q in queries:
            r=rq.get("https://api.search.brave.com/res/v1/web/search",params={"q":q,"count":max_per},
                headers={"Accept":"application/json","X-Subscription-Token":bk},timeout=15)
            diag.append({"p":"brave","q":q[:60],"s":r.status_code})
            if r.status_code==200:
                raw=r.text;sh=sha(raw)
                snaps.append({"query":q[:60],"hash":sh,"ts":datetime.now(timezone.utc).isoformat()})
                for res in r.json().get("web",{}).get("results",[]):
                    cands.append({"url":res.get("url",""),"title":res.get("title","")[:100],"provider":"brave","source_flag":"BRAVE_L2","snap_hash":sh})
    except:pass
    return cands,snaps
def _wikidata_ar0(cn,iso2,diag):
    results=[]
    terms=[f"Ministry of Education ({cn})",f"Ministry of Education {cn}",f"examination board {cn}",f"Education in {cn}"]
    try:
        import urllib.request as ul,urllib.parse as up
        for term in terms:
            su='https://en.wikipedia.org/w/api.php?'+up.urlencode({'action':'query','list':'search','srsearch':term,'format':'json','srlimit':3})
            req=ul.Request(su,headers={'User-Agent':f'SMAXIA/{VERSION}'})
            with ul.urlopen(req,timeout=12)as r:data=json.loads(r.read())
            for res in data.get('query',{}).get('search',[]):
                title=res['title']
                wu='https://en.wikipedia.org/w/api.php?'+up.urlencode({'action':'query','titles':title,'prop':'pageprops','ppprop':'wikibase_item','format':'json'})
                with ul.urlopen(ul.Request(wu,headers={'User-Agent':f'SMAXIA/{VERSION}'}),timeout=10)as r2:data2=json.loads(r2.read())
                for pid,page in data2.get('query',{}).get('pages',{}).items():
                    qid=page.get('pageprops',{}).get('wikibase_item')
                    if not qid:continue
                    wdu=f'https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={qid}&property=P856&format=json'
                    with ul.urlopen(ul.Request(wdu,headers={'User-Agent':f'SMAXIA/{VERSION}'}),timeout=10)as r3:data3=json.loads(r3.read())
                    for c2 in data3.get('claims',{}).get('P856',[]):
                        u=c2.get('mainsnak',{}).get('datavalue',{}).get('value','')
                        if not any(k in u.lower()for k in['.gov','.gouv','.edu','.ac.','.nic.']):continue
                        dom=urlparse(u).netloc.lower()
                        if f".{iso2.lower()}"not in dom and not any(k in dom for k in[".gov",".gouv",".edu"]):continue
                        if u not in[x["url"]for x in results]:
                            ie=any(k in title.lower()for k in["exam","board","council","certificate"])
                            results.append({"url":u,"title":title,"provider":"wikidata","is_exam":ie})
            if len(results)>=4:break
    except:pass
    results.sort(key=lambda x:x["url"]);return results
def _wikidata_lang(cn,iso2):
    """Wikidata P37 (official language) — NO hardcoded country lists."""
    try:
        import urllib.request as ul
        # Find country QID
        su=f'https://www.wikidata.org/w/api.php?action=wbsearchentities&search={cn}&language=en&type=item&limit=3&format=json'
        with ul.urlopen(ul.Request(su,headers={'User-Agent':f'SMAXIA/{VERSION}'}),timeout=10)as r:data=json.loads(r.read())
        for res in data.get('search',[]):
            qid=res['id']
            # Check P37 (official language)
            wu=f'https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={qid}&property=P37&format=json'
            with ul.urlopen(ul.Request(wu,headers={'User-Agent':f'SMAXIA/{VERSION}'}),timeout=10)as r2:data2=json.loads(r2.read())
            claims=data2.get('claims',{}).get('P37',[])
            if not claims:continue
            lang_qids=[c.get('mainsnak',{}).get('datavalue',{}).get('value',{}).get('id','')for c in claims]
            # Map language QIDs to ISO codes
            lang_map={"Q150":"fr","Q1860":"en","Q1321":"es","Q5146":"pt","Q13955":"ar","Q188":"de","Q652":"it","Q7850":"zh","Q9610":"bn","Q11059":"ru","Q1571":"hi"}
            langs=[]
            for lq in lang_qids:
                if lq in lang_map and lang_map[lq]not in langs:langs.append(lang_map[lq])
            if langs:
                if"en"not in langs:langs.append("en")
                return langs[:3]
    except:pass
    return["en"]
def bloc2_da0(seed):
    iso2=seed["iso2"];cn=seed["country_name"];log_ev("BLOC2_DA0",iso2)
    queries=[f'"ministry of education" "{cn}"',f'"examination board" "{cn}"',f'"national curriculum" "{cn}"',f'"secondary education" "{cn}"']
    diag=[];bcands,bsnaps=_brave_search(queries[:4],diag)
    wcands=_wikidata_ar0(cn,iso2,diag);cands=bcands+wcands
    langs=_wikidata_lang(cn,iso2);seed["languages"]=langs
    if not cands:return{"status":"FAIL","code":"NO_CANDIDATES","iso2":iso2,"country":cn,"diag":diag,"candidates":[],"selected":None,"domains":[],"brave_snaps":bsnaps,"languages":langs}
    seen=set();verified=[]
    for c in cands:
        u=c["url"];dom=urlparse(u).netloc
        if dom in seen:continue
        seen.add(dom);html,sc,_,_=http_get(u,timeout=12);sf=c.get("source_flag","")
        if sc not in range(200,300):
            ps=0.8 if any(k in dom for k in[".gov",".gouv"])else(0.65 if any(k in dom for k in[".edu",".ac."])else 0.3)
            verified.append({"url":u,"domain":dom,"score":ps,"accessible":False,"provider":c.get("provider",""),"is_exam":c.get("is_exam",False),"source_flag":sf})
        else:
            auth=_auth_score(u,html)
            verified.append({"url":u,"domain":dom,"score":auth["score"],"reasons":auth["reasons"],"accessible":True,"provider":c.get("provider",""),"is_exam":c.get("is_exam",False),"source_flag":sf})
    verified.sort(key=lambda x:(-x["score"],sha(x["url"])))
    ar0_edu=next((v for v in verified if v["score"]>=0.55 and not v.get("is_exam")),None)
    ar0_exam=next((v for v in verified if v.get("is_exam")and v["score"]>=0.55),None)
    if not ar0_edu:ar0_edu=next((v for v in verified if v["score"]>=0.55),None)
    if not ar0_exam and not ar0_edu:return{"status":"FAIL","code":"NO_AUTHORITY","iso2":iso2,"country":cn,"diag":diag,"candidates":verified[:20],"selected":None,"domains":[],"brave_snaps":bsnaps,"languages":langs}
    selected={"AR0_EDU":ar0_edu,"AR0_EXAM":ar0_exam}
    domains=list(set(v["domain"]for v in[ar0_edu,ar0_exam]if v))
    ar0={"status":"OK","iso2":iso2,"country":cn,"candidates":verified[:50],"selected":selected,"domains":domains,"diag":diag,"brave_snaps":bsnaps,"languages":langs}
    ar0["sha256"]=sha(cj(sv(ar0)));return ar0
# ═══ BLOC 3: CAP BUILD ═══
_LANG_TERMS={
    "fr":{"exam":["sujet","épreuve","annales"],"answer":["corrigé","correction","barème"],"diploma":["baccalauréat","brevet","concours"]},
    "en":{"exam":["exam","question paper","past paper"],"answer":["answer key","mark scheme","solution"],"diploma":["examination","certificate","board exam"]},
    "es":{"exam":["examen","prueba"],"answer":["solución","corrección"],"diploma":["bachillerato","selectividad"]},
    "pt":{"exam":["exame","prova"],"answer":["gabarito","correção"],"diploma":["ENEM","vestibular"]},
    "ar":{"exam":["امتحان","اختبار"],"answer":["تصحيح","حل"],"diploma":["بكالوريا","شهادة"]},
}
def _wiki_cap(cn):
    """Parse Wikipedia 'Education in {country}' for CAP structure."""
    cap_data={"levels":[],"subjects":[],"chapters":[],"exams":[],"contests":[],"coefs":{}}
    try:
        import urllib.request as ul,urllib.parse as up
        title=f"Education in {cn}"
        wu='https://en.wikipedia.org/w/api.php?'+up.urlencode({'action':'parse','page':title,'prop':'wikitext|sections','format':'json'})
        with ul.urlopen(ul.Request(wu,headers={'User-Agent':f'SMAXIA/{VERSION}'}),timeout=15)as r:data=json.loads(r.read())
        wt=data.get('parse',{}).get('wikitext',{}).get('*','')
        secs=data.get('parse',{}).get('sections',[])
        low=wt.lower()
        # Extract levels
        for pat,name in[(r'(?:secondary|lycée|high school|senior)',  "Secondary"),(r'(?:primary|école|elementary)',"Primary"),(r'(?:university|higher|supérieur)',"Higher")]:
            if re.search(pat,low):cap_data["levels"].append({"name":name,"source":"wikipedia"})
        # Extract exam names
        for pat,name in[(r'baccalaur[ée]at',"Baccalauréat"),(r'brevet',"Brevet"),(r'\bgce\b',"GCE"),(r'\bcbse\b',"CBSE"),(r'waec',"WAEC"),(r'concours',"Concours"),(r'\bgcse\b',"GCSE"),(r'entrance.exam',"Entrance Exam")]:
            if re.search(pat,low):cap_data["exams"].append({"name":name,"source":"wikipedia"})
    except:pass
    return cap_data
def bloc3_cap(ar0,seed):
    iso2=seed["iso2"];cn=seed["country_name"];langs=seed.get("languages",["en"])
    log_ev("BLOC3_CAP",iso2)
    if ar0["status"]!="OK":return{"status":"FAIL","code":"AR0_FAIL","iso2":iso2,"sections":{}}
    domains=ar0.get("domains",[])
    # A) Wikipedia structure
    wiki=_wiki_cap(cn)
    # B) Crawl AR0 for PDFs and structure
    pages,pdfs=[],[];allowed=set(domains)
    seeds=[v["url"]for v in[ar0["selected"].get("AR0_EDU"),ar0["selected"].get("AR0_EXAM")]if v]
    exam_dom=ar0["selected"].get("AR0_EXAM")
    if exam_dom and exam_dom.get("accessible"):
        base=exam_dom["url"].rstrip("/")
        for ep in["/question-paper","/past-papers","/examinations","/annales","/sujets","/archives","/downloads"]:seeds.append(base+ep)
    visited=set()
    for url in seeds:
        if url in visited:continue
        visited.add(url);html,sc,_,_=http_get(url,timeout=15)
        if sc not in range(200,300):continue
        low=(html or"").lower()[:5000]
        role="EXAM_ARCHIVE"if any(k in low for k in["past paper","question paper","annale","sujet","archive"])else("SYLLABUS"if any(k in low for k in["syllabus","curriculum","programme"])else"LANDING")
        pages.append({"url":url,"domain":urlparse(url).netloc,"role":role})
        for link in _links(html,url):
            lu=link["url"];ldom=urlparse(lu).netloc
            if".pdf"in lu.lower()and(ldom in allowed or _dom_score(ldom)>=0.5 or _is_sub(ldom,allowed)):
                pdfs.append({"url":lu,"text":link["text"][:80],"domain":ldom})
        for link in _links(html,url)[:40]:
            lu,lt=link["url"],link["text"].lower();ldom=urlparse(lu).netloc
            if lu in visited or".pdf"in lu.lower():continue
            if any(k in lt or k in lu.lower()for k in["exam","paper","syllabus","programme","sujet","annale","archive"])and(ldom in allowed or _dom_score(ldom)>=0.7):
                visited.add(lu);h2,s2,_,_=http_get(lu,timeout=12)
                if s2 in range(200,300):
                    pages.append({"url":lu,"domain":ldom,"role":"CRAWLED"})
                    for pl in _links(h2,lu,lambda u,t:".pdf"in u.lower())[:20]:
                        pd=urlparse(pl["url"]).netloc
                        if pd in allowed or _dom_score(pd)>=0.5:pdfs.append({"url":pl["url"],"text":pl["text"][:80],"domain":pd})
            if len(visited)>25:break
    # Detect subjects from PDFs
    blob=" ".join([d.get("text","")+d["url"]for d in pdfs]).lower()
    subjects=[{"name":s,"detected":True}for s in["Mathematics","Physics","Chemistry","Biology","Literature","History","Geography","Philosophy","Economics","English","French","Computer Science"]if s.lower()in blob]
    # Detect grading
    grading={"max":0,"description":"[NOT_DETECTED]"}
    for p in pages:
        html,_,_,_=http_get(p["url"],cache_only=True)
        if html and re.search(r'(?:note|barème).*?/\s*20\b',html.lower()):grading={"max":20,"description":"Scale 0-20"};break
        if html and re.search(r'(?:maximum|total)\s+marks?\s*:?\s*100',html.lower()):grading={"max":100,"description":"Scale 0-100"};break
    status="DISCOVERED"if pdfs else("PARTIAL"if pages else"FAIL")
    cap={"A_METADATA":{"cap_id":f"CAP_{iso2}_{sha(cn)[:8]}","iso2":iso2,"country":cn,"languages":langs,"status":status},
         "B_EDUCATION":{"levels":wiki["levels"],"subjects":subjects,"chapters":[],"chapters_status":"PENDING_EXTRACTION"},
         "C_HARVEST":{"authority_domains":domains,"pages_crawled":len(pages),"pdfs_found":len(pdfs),"documents":pdfs[:100],"authority_graph":pages[:50]},
         "D_KERNEL":{"atom_min_len":40,"cluster_min":2,"qc_target":15,"coverage_target":1.0},
         "E_EXAMS":{"exams":wiki["exams"][:7],"contests":wiki["contests"][:7],"coefs":wiki["coefs"],"grading":grading}}
    cap["sha256"]=sha(cj(sv(cap)));log_ev("CAP_END",f"{status} {len(pdfs)}pdf");return cap
# ═══ BLOC 4: DA1 Harvest ═══
def _query_pack(cn,domains,langs):
    qs=[];year=datetime.now().year
    for dom in domains[:3]:
        for lang in langs:
            t=_LANG_TERMS.get(lang,_LANG_TERMS["en"])
            eo=" OR ".join(f'"{x}"'for x in t["exam"][:2]);ao=" OR ".join(f'"{x}"'for x in t["answer"][:2])
            qs+=[f"site:{dom} filetype:pdf ({eo}) {year}",f"site:{dom} filetype:pdf ({eo}) {year-1}",f"site:{dom} filetype:pdf ({ao})"]
    for lang in langs[:2]:
        t=_LANG_TERMS.get(lang,_LANG_TERMS["en"])
        qs+=[f"{cn} filetype:pdf {t['exam'][0]} {t['diploma'][0]} {year}",f"{cn} filetype:pdf {t['answer'][0]} {year}"]
    return qs[:20]
def _clf_pdf(url,text):
    low=(url+" "+text).lower()
    return"corrige"if any(k in low for k in["corrig","correction","answer","solution","bareme","mark scheme","barème"])else"sujet"
def _clf_type(url,text):
    low=(url+" "+text).lower()
    if any(k in low for k in["concours","competitive","entrance"]):return"CONCOURS"
    if any(k in low for k in["dst","devoir","controle","test","interro"]):return"DST"
    if any(k in low for k in["exercice","exercise","td","travaux"]):return"EXERCISE"
    return"EXAM"
def bloc4_da1(ar0,cap):
    cn=cap["A_METADATA"]["country"];langs=cap["A_METADATA"]["languages"];domains=cap["C_HARVEST"]["authority_domains"]
    log_ev("BLOC4_DA1")
    manifest,diag=[],[]
    # Layer 1: CAP crawl PDFs
    for d in cap["C_HARVEST"].get("documents",[]):
        manifest.append({"url":d["url"],"role":_clf_pdf(d["url"],d.get("text","")),"exam_type":_clf_type(d["url"],d.get("text","")),"hash":sha(d["url"])[:32],"domain":d.get("domain",""),"text":d.get("text","")[:80],"provider":"ar0_crawl","source_flag":"AR0_CRAWL","auth_score":_dom_score(d.get("domain",""))})
    # Layer 2: Brave Search
    bsnaps=[]
    if len(manifest)<10 and _secret("BRAVE_KEY"):
        queries=_query_pack(cn,domains,langs);bcands,bsnaps=_brave_search(queries[:8],diag)
        for r2 in bcands:
            if".pdf"in r2["url"].lower():
                dom=urlparse(r2["url"]).netloc
                if _is_sub(dom,domains)or _dom_score(dom)>=0.5 or any(k in r2.get("text","").lower()for k in["sujet","exam","paper","corrigé","annale"]):
                    manifest.append({"url":r2["url"],"role":_clf_pdf(r2["url"],r2.get("text","")),"exam_type":_clf_type(r2["url"],r2.get("text","")),"hash":sha(r2["url"])[:32],"domain":dom,"text":r2.get("text","")[:80],"provider":"brave","source_flag":"BRAVE_L2","auth_score":_dom_score(dom)})
    # Dedupe
    seen=set();unique=[]
    for p in manifest:
        if p["url"]not in seen:seen.add(p["url"]);unique.append(p)
    manifest=unique[:30]
    mode="REAL"if manifest else"FAIL_DA1_NO_SOURCE"
    # Pairing
    suj=[s for s in manifest if s["role"]=="sujet"];cor=[s for s in manifest if s["role"]=="corrige"]
    pairs,quar,unpaired=[],[],[];used=set()
    for s in suj:
        cands=[]
        for j,c in enumerate(cor):
            if j in used:continue
            sc=0;sl,cl=s["url"].lower(),c["url"].lower()
            if"/".join(sl.split("/")[:-1])=="/".join(cl.split("/")[:-1]):sc+=3
            if urlparse(sl).netloc==urlparse(cl).netloc:sc+=1
            sy=re.findall(r'20\d{2}',sl);cy=re.findall(r'20\d{2}',cl)
            if sy and cy and set(sy)&set(cy):sc+=2
            sc+=min(len(set(re.findall(r'[a-z]{4,}',sl))&set(re.findall(r'[a-z]{4,}',cl))),3)
            cands.append((j,sc))
        cands.sort(key=lambda x:-x[1])
        if len(cands)>=2 and cands[0][1]>0 and cands[0][1]==cands[1][1]:
            quar.append({"url":s["url"],"reason":"QUARANTINE_PAIRING_AMBIGUOUS"});continue
        if cands and cands[0][1]>=1:
            j,bs=cands[0];c=cor[j];used.add(j)
            pairs.append({"pair_id":f"P_{len(pairs):04d}","sujet_url":s["url"],"corrige_url":c["url"],"sujet_hash":s["hash"],"corrige_hash":c["hash"],"score":bs,"mode":"FULL","exam_type":s["exam_type"],"source_flag":s.get("source_flag","")})
        else:
            pairs.append({"pair_id":f"P_{len(pairs):04d}","sujet_url":s["url"],"corrige_url":"","sujet_hash":s["hash"],"corrige_hash":"","score":0,"mode":"SUBJECT_ONLY","exam_type":s["exam_type"],"source_flag":s.get("source_flag","")})
    for j,c in enumerate(cor):
        if j not in used:unpaired.append({"url":c["url"],"reason":"no_sujet_match"})
    return{"mode":mode,"manifest":manifest,"pairs":pairs,"quarantine":quar,"unpaired":unpaired,"brave_snaps":bsnaps,"diag":diag,"stats":{"total":len(manifest),"sujets":len(suj),"corriges":len(cor),"pairs":len(pairs),"quarantined":len(quar)}}
# ═══ BLOC 5: ATOMISATION ═══
def _ocr(pdf_bytes):
    if not pdf_bytes:return"","no_data"
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes))as pdf:
            txt="\n\n".join(p.extract_text()or""for p in pdf.pages[:15])
            if len(txt.strip())>50:return txt.strip(),"pdfplumber"
    except:pass
    return"","extraction_failed"
_HP=[r'roll\s*no',r'candidates?\s+must',r'time\s*(?:allowed|allotted)',r'maximum\s+marks',r'ne\s+rien\s+écrire',r'page\s+\d+\s+(?:of|de|sur)',r'P\.?\s*T\.?\s*O\.?']
_IP=[r'answer\s+any\s+\d+',r'attempt\s+(?:all|any)',r'instructions?\s*:',r'general\s+instructions']
def _is_hdr(text):
    low=text.lower()
    return sum(1 for p in _HP if re.search(p,low))>=2 or(sum(1 for p in _IP if re.search(p,low))>=1 and not re.search(r'(?:what|why|how|explain|calculate|find|solve|prove)',low))
def _split_q(text):
    if not text or len(text)<100:return[]
    ms=list(re.finditer(r'(?:^|\n)\s*(?:Q\.?\s*)?(\d{1,3})\s*[\.\)\-:]\s*(.+?)(?=(?:^|\n)\s*(?:Q\.?\s*)?\d{1,3}\s*[\.\)\-:]|\Z)',text,re.DOTALL|re.MULTILINE))
    if len(ms)>=3:
        parts=[m.group(0).strip()for m in ms if len(m.group(0).strip())>=40 and not _is_hdr(m.group(0))]
        if len(parts)>=2:return parts
    for pat in[r'(?=(?:Exercice|EXERCICE|Exercise|SECTION|PARTIE)\s*[-:.]?\s*[A-Za-z0-9IViv])',r'(?=(?:QUESTION|Question)\s*\d)']:
        parts=[p.strip()for p in re.split(pat,text,flags=re.I)if len(p.strip())>=60 and not _is_hdr(p)]
        if len(parts)>=2:return parts
    parts=[p.strip()for p in text.split("\n\n")if len(p.strip())>=60 and not _is_hdr(p)]
    if len(parts)>=3:return parts
    if len(text)>=200 and not _is_hdr(text):return[text]
    return[]
def _pts(text):
    for pat in[r'[\(\[]\s*(\d{1,3})\s*(?:marks?|points?|pts?)\s*[\)\]]',r'(\d{1,3})\s*(?:marks?|points?)\b',r'/\s*(\d{1,3})\b']:
        m=re.search(pat,text,re.I)
        if m:
            val=int([g for g in m.groups()if g][-1])
            if 1<=val<=200:return val
    return None
def bloc5_atoms(da1):
    log_ev("BLOC5_ATOMS");atoms=[]
    for pair in da1["pairs"][:15]:
        pid=pair["pair_id"];sd,ss,_,_=http_get(pair["sujet_url"],binary=True,timeout=20)
        cd,cs=None,-1
        if pair["corrige_url"]:cd,cs,_,_=http_get(pair["corrige_url"],binary=True,timeout=20)
        st_,sm=_ocr(sd if isinstance(sd,bytes)else None);ct_,cm2=_ocr(cd if isinstance(cd,bytes)else None)
        if not st_ or len(st_)<100:continue
        qb=_split_q(st_);cb=_split_q(ct_)if ct_ else[]
        has_nums=any(re.match(r'\s*\d+[\.\)]',p)for p in qb)
        conf=0.9 if has_nums and len(qb)>=3 else(0.6 if has_nums else 0.4)
        qi_c=0
        for qi_idx,q in enumerate(qb):
            low=q.lower()
            if _is_hdr(q):continue
            is_q=any(k in low for k in["what","why","how","explain","calculate","find","solve","prove","define","describe","qu'est","expliquer","calculer","démontrer","?","marks","points"])or len(q)>=150
            if not is_q:continue
            qi_c+=1;rb=cb[qi_idx]if qi_idx<len(cb)else""
            posable_corrige=len(rb)>50;posable_eval=is_q
            atoms.append({"qi_id":f"{pid}_Q{qi_c}","pair_id":pid,"qi_text":q.strip(),"rqi_text":rb.strip(),"points":_pts(q),
                "qi_hash":sha(q.strip()),"rqi_hash":sha(rb.strip()),"confidence":round(conf,2),
                "exam_type":pair.get("exam_type","EXAM"),"source_flag":pair.get("source_flag",""),
                "posable":{"corrige":posable_corrige,"scope":True,"evaluable":posable_eval,"final":posable_corrige and posable_eval},
                "chapter_code":"PENDING"})
    log_ev("ATOMS_END",f"{len(atoms)}");return atoms
# ═══ BLOC 6: IA1_MINER ═══
def bloc6_miner(atoms):
    log_ev("BLOC6_MINER");traces=[]
    for a in atoms:
        if not a["posable"]["final"]:continue
        rqi=a["rqi_text"];steps=[]
        # Rule-based step extraction from RQi
        parts=re.split(r'\n+',rqi)
        for i,p in enumerate(parts):
            p=p.strip()
            if len(p)<20:continue
            # Type each step
            low=p.lower()
            if any(k in low for k in["donc","thus","therefore","alors","d'où","hence"]):stype="CONCLUSION"
            elif any(k in low for k in["calcul","compute","=","solve","résoudre"]):stype="COMPUTATION"
            elif any(k in low for k in["appliq","apply","using","utilisant","formule"]):stype="APPLICATION"
            elif any(k in low for k in["définit","define","soit","let","posons"]):stype="DEFINITION"
            else:stype="REASONING"
            steps.append({"step":i+1,"text":p[:200],"type":stype})
        # SIG(q) = <Action, Property, Object, conteXt>
        qi_low=a["qi_text"].lower()
        action="CALCULATE"if any(k in qi_low for k in["calcul","compute","find","trouv"])else("PROVE"if any(k in qi_low for k in["prove","démontrer","montrer","show"])else("EXPLAIN"if any(k in qi_low for k in["explain","expliquer","justifier"])else"DETERMINE"))
        sig_q={"A":action,"P":"","O":"","X":""}
        traces.append({"qi_id":a["qi_id"],"steps":steps,"sig_q":sig_q,"n_steps":len(steps)})
    log_ev("MINER_END",f"{len(traces)}");return traces
# ═══ BLOC 7: CLUSTERING ═══
def bloc7_clustering(atoms,traces):
    log_ev("BLOC7_CLUSTER")
    # Group by chapter_code + SIG action
    trace_map={t["qi_id"]:t for t in traces}
    groups={};quarantine=[]
    for a in atoms:
        if not a["posable"]["final"]:continue
        t=trace_map.get(a["qi_id"])
        if not t:continue
        key=f"{a['chapter_code']}_{t['sig_q']['A']}"
        groups.setdefault(key,[]).append(a["qi_id"])
    clusters=[];singletons=[]
    for key,qids in sorted(groups.items()):
        if len(qids)>=2:clusters.append({"cluster_id":f"CL_{len(clusters):04d}","key":key,"qi_ids":qids,"n":len(qids)})
        else:singletons.append({"qi_id":qids[0],"reason":"SINGLETON","key":key})
    log_ev("CLUSTER_END",f"{len(clusters)}cl {len(singletons)}sing");return{"clusters":clusters,"singletons":singletons}
# ═══ BLOC 8: IA1_BUILDER ═══
def bloc8_builder(clusters,atoms,traces):
    log_ev("BLOC8_BUILDER");atom_map={a["qi_id"]:a for a in atoms};trace_map={t["qi_id"]:t for t in traces}
    qc_list=[]
    for cl in clusters["clusters"]:
        qids=cl["qi_ids"];reps=[atom_map[q]for q in qids if q in atom_map]
        if not reps:continue
        # QC_text: derived from representative Qi (no template)
        rep=reps[0];qi_text=rep["qi_text"][:200]
        qm=re.search(r'[^.!?]*\?',qi_text)
        qc_text=qm.group(0).strip()if qm else qi_text[:120]
        # ARI from traces consensus
        all_steps=[];all_types=[]
        for q in qids:
            t=trace_map.get(q)
            if t:all_steps.extend(t["steps"]);all_types.extend([s["type"]for s in t["steps"]])
        type_freq={};
        for tt in all_types:type_freq[tt]=type_freq.get(tt,0)+1
        ari={"steps_consensus":sorted(type_freq.items(),key=lambda x:-x[1])[:5],"n_steps_avg":round(len(all_steps)/max(len(qids),1),1)}
        # FRT: 4 blocks from RQi
        all_rqi=" ".join([atom_map[q]["rqi_text"]for q in qids if q in atom_map and atom_map[q]["rqi_text"]])[:2000]
        frt={"usage":all_rqi[:200]if all_rqi else"","reponse_type":"analytical"if any(k in all_rqi.lower()for k in["calcul","=","résult"])else"descriptive",
             "pieges":[],"conclusion":all_rqi[-200:]if len(all_rqi)>200 else""}
        # Triggers
        triggers=[]
        for q in qids:
            t=trace_map.get(q)
            if t:
                for s in t["steps"][:3]:triggers.append({"text":s["text"][:80],"type":s["type"],"from":q})
        triggers=triggers[:7]
        qc_list.append({"qc_id":f"QC_{len(qc_list):04d}","cluster_id":cl["cluster_id"],"qc_text":qc_text,
            "qi_ids":qids,"ari":ari,"frt":frt,"triggers":triggers,"n_triggers":len(triggers),
            "chapter":cl["key"].split("_")[0]if"_"in cl["key"]else cl["key"]})
    log_ev("BUILDER_END",f"{len(qc_list)}");return qc_list
# ═══ BLOC 9: IA2_JUDGE ═══
def bloc9_judge(qc_list,atoms):
    log_ev("BLOC9_JUDGE");results=[]
    for qc in qc_list:
        checks={"CHK_POSABLE_VALID":len(qc["qi_ids"])>=2,
                 "CHK_QC_FORM":len(qc["qc_text"])>=20,
                 "CHK_FRT_OK":bool(qc["frt"].get("usage")),
                 "CHK_TRIGGERS_QUALITY":qc["n_triggers"]>=1,
                 "CHK_ARI_TYPED":len(qc["ari"].get("steps_consensus",[]))>=1,
                 "CHK_ANTI_SINGLETON":len(qc["qi_ids"])>=2}
        verdict="PASS"if all(checks.values())else"FAIL"
        results.append({"qc_id":qc["qc_id"],"verdict":verdict,"checks":checks})
    log_ev("JUDGE_END",f"{sum(1 for r in results if r['verdict']=='PASS')}PASS");return results
# ═══ BLOC 10: F1/F2 ═══
def bloc10_scoring(atoms,qc_list,judge):
    log_ev("BLOC10_F1F2")
    passed={r["qc_id"]for r in judge if r["verdict"]=="PASS"}
    valid_qc=[q for q in qc_list if q["qc_id"]in passed]
    posable=[a for a in atoms if a["posable"]["final"]]
    if not valid_qc or not posable:return{"f1":{"score":0,"status":"NO_DATA"},"f2":{"score":0,"status":"NO_DATA"},"granulo_k":[],"selected_qc":[]}
    n_qc=len(valid_qc);n_posable=len(posable);n_atoms=len(atoms)
    cov=len(set(qi for q in valid_qc for qi in q["qi_ids"]))/max(n_posable,1)
    div=min(n_qc/15,1.0)
    f1=round(min(cov*0.6+div*0.4,1.0),4)
    f2=round(f1*0.85,4)
    return{"f1":{"score":f1,"status":"OK","formula_mode":"TEST_CLEAR","n_qc":n_qc,"coverage":round(cov,4)},
           "f2":{"score":f2,"status":"OK","formula_mode":"TEST_CLEAR","predicted_range":[round(f2-0.05,4),round(min(f2+0.05,1),4)]},
           "granulo_k":valid_qc[:15],"selected_qc":[q["qc_id"]for q in valid_qc[:15]]}
# ═══ BLOC 11: COVERAGE + SEAL ═══
def bloc11_coverage(atoms,qc_list,judge,scoring):
    log_ev("BLOC11_COVERAGE")
    passed={r["qc_id"]for r in judge if r["verdict"]=="PASS"}
    valid_qc=[q for q in qc_list if q["qc_id"]in passed]
    posable=[a for a in atoms if a["posable"]["final"]]
    # Map each posable Qi to its QC mère
    qc_map={};
    for q in valid_qc:
        for qi in q["qi_ids"]:qc_map[qi]=q["qc_id"]
    covered,orphans=[],[]
    for a in posable:
        if a["qi_id"]in qc_map:covered.append({"qi_id":a["qi_id"],"qc_id":qc_map[a["qi_id"]]})
        else:orphans.append({"qi_id":a["qi_id"],"reason":"NO_QC_MOTHER"})
    rate=len(covered)/max(len(posable),1)
    sealed=rate>=1.0
    # Saturation check
    saturated=len(orphans)==0 and len(valid_qc)>=2
    return{"coverage_rate":round(rate,4),"covered":len(covered),"orphans_count":len(orphans),"orphans":orphans[:20],
           "posable_total":len(posable),"sealed":sealed,"saturated":saturated,
           "seal_hash":sha(cj(sv({"qc":[q["qc_id"]for q in valid_qc],"cov":round(rate,4)})))if sealed else None}
# ═══ PIPELINE ORCHESTRATOR ═══
def run_full_pipeline(iso2,_replay=False):
    st.session_state["_sil"]=_replay;log_ev("PIPELINE",iso2)
    seed=bloc1_activate(iso2)
    if seed["status"]!="OK":st.session_state["_sil"]=False;return{"error":f"GATE_ISO2_VALID FAIL: {iso2}"}
    ar0=bloc2_da0(seed)
    cap=bloc3_cap(ar0,seed)
    da1=bloc4_da1(ar0,cap)
    atoms=bloc5_atoms(da1)
    traces=bloc6_miner(atoms)
    clusters=bloc7_clustering(atoms,traces)
    qc_list=bloc8_builder(clusters,atoms,traces)
    judge=bloc9_judge(qc_list,atoms)
    scoring=bloc10_scoring(atoms,qc_list,judge)
    coverage=bloc11_coverage(atoms,qc_list,judge,scoring)
    # Gates
    gates=[]
    def G(n,c,e):gates.append({"gate":n,"verdict":"PASS"if c else"FAIL","evidence":e})
    G("GATE_ISO2_VALID",seed["status"]=="OK",iso2)
    G("GATE_AR0_MIN",ar0.get("status")=="OK" and len(ar0.get("domains",[]))>=1,f"doms={len(ar0.get('domains',[]))}")
    G("GATE_CAP_SCHEMA",cap.get("A_METADATA",{}).get("status") in["DISCOVERED","PARTIAL"],cap.get("A_METADATA",{}).get("status",""))
    G("GATE_DA1_MIN",len(da1.get("pairs",[]))>=1,f"pairs={len(da1.get('pairs',[]))}")
    G("GATE_ATOMS_MIN",len([a for a in atoms if a["posable"]["final"]])>=1,f"posable={len([a for a in atoms if a['posable']['final']])}")
    G("GATE_MINER",len(traces)>=1,f"traces={len(traces)}")
    G("GATE_CLUSTERS",len(clusters.get("clusters",[]))>=1,f"clusters={len(clusters.get('clusters',[]))}")
    G("GATE_QC_BUILT",len(qc_list)>=1,f"qc={len(qc_list)}")
    G("GATE_IA2",sum(1 for r in judge if r["verdict"]=="PASS")>=1,f"pass={sum(1 for r in judge if r['verdict']=='PASS')}")
    G("GATE_F1",scoring["f1"].get("status")=="OK",f"f1={scoring['f1'].get('score',0):.4f}")
    G("GATE_COVERAGE",coverage["coverage_rate"]>=0.5,f"cov={coverage['coverage_rate']:.1%}")
    global_v="PASS"if all(g["verdict"]=="PASS"for g in gates)else"FAIL"
    rid=f"RUN_{iso2}_{sha(cj(sv(cap)))[:8]}"
    # Artifacts
    al=[("AR0",ar0),("CAP",cap),("DA1_manifest",da1["manifest"]),("DA1_pairs",da1["pairs"]),("BraveSnaps",da1.get("brave_snaps",[])),
        ("Atoms",atoms),("Traces",traces),("Clusters",clusters),("QC",qc_list),("Judge",judge),("F1F2",scoring),("Coverage",coverage),
        ("Gates",{"gates":gates,"global":global_v}),("UI_LOG",st.session_state.get("ui_ev",[]))]
    arts={n:w_art(rid,n,d)for n,d in al}
    seal={"run_id":rid,"iso2":iso2,"version":VERSION,"verdict":global_v,"f1":scoring["f1"].get("score",0),"f2":scoring["f2"].get("score",0),
          "n_atoms":len(atoms),"n_posable":len([a for a in atoms if a["posable"]["final"]]),"n_qc":len(qc_list),"n_qc_pass":sum(1 for r in judge if r["verdict"]=="PASS"),
          "n_pairs":len(da1["pairs"]),"coverage":coverage["coverage_rate"],"sealed":coverage["sealed"],"n_arts":len(arts)+1}
    arts["SealReport"]=w_art(rid,"SealReport",seal)
    st.session_state["_sil"]=False
    return{"run_id":rid,"seed":seed,"ar0":ar0,"cap":cap,"da1":da1,"atoms":atoms,"traces":traces,"clusters":clusters,"qc_list":qc_list,"judge":judge,"scoring":scoring,"coverage":coverage,"gates":{"gates":gates,"global":global_v},"seal":seal,"artifacts":arts}
# ═══ UI: MISSION CONTROL ═══
def main():
    st.set_page_config(page_title=f"SMAXIA v{VERSION}",layout="wide",initial_sidebar_state="collapsed")
    if"pipeline"not in st.session_state:st.session_state.pipeline=None
    if"ui_ev"not in st.session_state:st.session_state.ui_ev=[]
    # Top bar
    hdr=st.container()
    with hdr:
        c1,c2,c3=st.columns([4,2,2])
        with c1:st.markdown(f"## 🚀 SMAXIA MISSION CONTROL v{VERSION}")
        with c2:
            q=st.text_input("Country",key="cq",placeholder="FR, IN, SN...",label_visibility="collapsed")
            hits=typeahead(q);cc=None
            if hits:
                labels=[f"{e['n']} ({e['c']})"for e in hits]
                sel=st.selectbox("sel",options=labels,index=0,label_visibility="collapsed")
                cc=hits[labels.index(sel)]["c"]
            elif q and len(q.strip())==2:cc=q.strip().upper()
        with c3:
            if st.button("🚀 ACTIVATE",type="primary",use_container_width=True):
                if cc:
                    log_ev("ACTIVATE",cc)
                    with st.spinner(f"Pipeline {cc}..."):st.session_state.pipeline=run_full_pipeline(cc)
    # Status bar
    p=st.session_state.pipeline
    if p and"seal"in p:
        s=p["seal"]
        cols=st.columns(8)
        cols[0].metric("Country",s["iso2"]);cols[1].metric("Verdict",s["verdict"])
        cols[2].metric("F1",f"{s['f1']:.4f}");cols[3].metric("F2",f"{s['f2']:.4f}")
        cols[4].metric("Atoms",s["n_atoms"]);cols[5].metric("QC",f"{s['n_qc_pass']}/{s['n_qc']}")
        cols[6].metric("Coverage",f"{s['coverage']:.0%}");cols[7].metric("Sealed","✅"if s["sealed"]else"❌")
    # API Keys status
    with st.sidebar:
        st.markdown("**API Keys**")
        for k,e in[("Brave","BRAVE_KEY"),("SerpAPI","SERPAPI_KEY")]:st.write(f"{'✅'if _secret(e)else'❌'} {k}")
        if p and"seal"in p:
            st.divider();st.json(p["seal"])
    # 6 Tabs
    T=st.tabs(["📚 CAP","📄 SUJETS","🔬 Qi/RQi","🎯 QC","📊 KPI","🗺️ MAPPING"])
    # ═══ TAB 1: CAP ═══
    with T[0]:
        if not p:st.info("🚀 Activate a country to begin.");return
        cap=p["cap"]
        if cap.get("A_METADATA",{}).get("status")=="FAIL":st.error(f"CAP FAIL: {cap.get('code','')}");return
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("#### Niveaux")
            for lv in cap.get("B_EDUCATION",{}).get("levels",[]):st.write(f"▸ {lv['name']}")
            if not cap.get("B_EDUCATION",{}).get("levels"):st.caption("Extraction pending")
        with c2:
            st.markdown("#### Matières")
            for sub in cap.get("B_EDUCATION",{}).get("subjects",[]):st.write(f"▸ {sub['name']}")
        with c3:
            st.markdown("#### Spécialités")
            st.caption(f"Chapters: {cap.get('B_EDUCATION',{}).get('chapters_status','PENDING')}")
        st.divider()
        st.markdown("#### Examens & Concours")
        for ex in cap.get("E_EXAMS",{}).get("exams",[]):st.write(f"▸ {ex['name']}")
        st.markdown(f"**Grading:** {cap.get('E_EXAMS',{}).get('grading',{}).get('description','N/A')}")
        st.markdown(f"**Domains:** {cap.get('C_HARVEST',{}).get('authority_domains',[])}")
        st.markdown(f"**Pages:** {cap.get('C_HARVEST',{}).get('pages_crawled',0)} | **PDFs:** {cap.get('C_HARVEST',{}).get('pdfs_found',0)}")
    # ═══ TAB 2: SUJETS ═══
    with T[1]:
        if not p:st.info("Activate a country.");return
        da1=p["da1"]
        st.markdown(f"#### Harvest — {da1['mode']} ({da1['stats']['total']} sources, {da1['stats']['pairs']} pairs)")
        # Filters
        types=sorted(set(pr.get("exam_type","")for pr in da1["pairs"]))
        fc1,fc2=st.columns(2)
        with fc1:ftype=st.selectbox("Type",["ALL"]+types,key="ft")
        with fc2:fmode=st.selectbox("Mode",["ALL","FULL","SUBJECT_ONLY"],key="fm")
        for pr in da1["pairs"]:
            if ftype!="ALL"and pr.get("exam_type")!=ftype:continue
            if fmode!="ALL"and pr.get("mode")!=fmode:continue
            c1,c2,c3,c4,c5=st.columns([1,2,2,1,1])
            c1.write(pr["pair_id"]);c2.write(pr.get("exam_type",""));c3.write(pr["sujet_url"][:50]+"...")
            c4.write("✅"if pr["mode"]=="FULL"else"❌");c5.write(pr.get("source_flag",""))
        if da1.get("quarantine"):
            st.subheader("⚠️ Quarantine");[st.write(f"{q['reason']} — {q['url'][:60]}")for q in da1["quarantine"]]
        # Brave snapshots
        bs=da1.get("brave_snaps",[])
        if bs:st.subheader(f"🔒 Brave Snapshots ({len(bs)})");[st.caption(f"`{s['hash'][:16]}` {s['query']}")for s in bs]
    # ═══ TAB 3: Qi/RQi ═══
    with T[2]:
        if not p:st.info("Activate a country.");return
        atoms=p["atoms"];posable=[a for a in atoms if a["posable"]["final"]]
        st.markdown(f"#### Atoms: {len(atoms)} total, {len(posable)} POSABLE")
        fp=st.selectbox("Filter",["ALL","POSABLE","NOT_POSABLE"],key="fp")
        shown=atoms if fp=="ALL"else(posable if fp=="POSABLE"else[a for a in atoms if not a["posable"]["final"]])
        for a in shown[:50]:
            with st.expander(f"**{a['qi_id']}** | {a.get('exam_type','')} | {'✅ POSABLE'if a['posable']['final']else'❌'} | {a.get('points','?')}pts"):
                st.markdown(f"**Qi:** {a['qi_text'][:400]}")
                if a["rqi_text"]:st.markdown(f"**RQi:** {a['rqi_text'][:400]}")
                st.caption(f"Chapter: {a['chapter_code']} | Conf: {a['confidence']} | Hash: {a['qi_hash'][:12]}")
    # ═══ TAB 4: QC ═══
    with T[3]:
        if not p:st.info("Activate a country.");return
        qc_list=p["qc_list"];judge=p["judge"]
        judge_map={r["qc_id"]:r for r in judge}
        n_pass=sum(1 for r in judge if r["verdict"]=="PASS")
        st.markdown(f"#### Questions Clés: {n_pass} PASS / {len(qc_list)} total (target ~15)")
        for qc in qc_list:
            j=judge_map.get(qc["qc_id"],{})
            v=j.get("verdict","?")
            with st.expander(f"{'✅'if v=='PASS'else'❌'} **{qc['qc_id']}** — \"{qc['qc_text'][:80]}\""):
                st.write(f"**Qi associées:** {', '.join(qc['qi_ids'])}")
                st.write(f"**Triggers:** {qc['n_triggers']}")
                if qc.get("ari"):st.write(f"**ARI:** {qc['ari'].get('steps_consensus',[])} avg_steps={qc['ari'].get('n_steps_avg',0)}")
                if qc.get("frt"):st.write(f"**FRT type:** {qc['frt'].get('reponse_type','')}")
                if j.get("checks"):
                    st.caption("Checks: "+" ".join([f"{'✓'if v2 else'✗'}{k}"for k,v2 in j["checks"].items()]))
    # ═══ TAB 5: KPI ═══
    with T[4]:
        if not p:st.info("Activate a country.");return
        cov=p["coverage"];scoring=p["scoring"]
        st.markdown("#### SATURATION DASHBOARD")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("QC trouvées",f"{scoring['f1'].get('n_qc',0)} / ~15")
        c2.metric("Couverture",f"{cov['covered']}/{cov['posable_total']}")
        c3.metric("Orphelins",cov["orphans_count"])
        c4.metric("Saturé","✅ OUI"if cov["saturated"]else"❌ NON")
        # Coverage bar
        rate=cov["coverage_rate"]
        st.progress(min(rate,1.0),text=f"Coverage: {rate:.1%}")
        # Gates
        st.markdown("#### Gates Pipeline")
        for g in p["gates"]["gates"]:st.write(f"{'✅'if g['verdict']=='PASS'else'🔴'} **{g['gate']}** `{g['evidence']}`")
        st.markdown(f"**GLOBAL: {p['gates']['global']}**")
        # Orphans detail
        if cov["orphans"]:
            st.markdown("#### Orphelins (Qi sans QC mère)")
            for o in cov["orphans"][:10]:st.write(f"⚠️ {o['qi_id']}")
        # F1/F2
        st.divider();fc1,fc2=st.columns(2)
        with fc1:st.metric("F1",f"{scoring['f1'].get('score',0):.4f}");st.caption(scoring["f1"].get("formula_mode",""))
        with fc2:st.metric("F2",f"{scoring['f2'].get('score',0):.4f}");rg=scoring["f2"].get("predicted_range",[0,0]);st.caption(f"Range: [{rg[0]:.4f}—{rg[1]:.4f}]")
    # ═══ TAB 6: MAPPING ═══
    with T[5]:
        if not p:st.info("Activate a country.");return
        st.markdown("#### 📎 Test de Couverture — Upload un sujet PDF")
        uploaded=st.file_uploader("Upload sujet PDF",type=["pdf"],key="map_pdf")
        if uploaded:
            pdf_bytes=uploaded.read()
            txt,mode=_ocr(pdf_bytes)
            if txt:
                qb=_split_q(txt)
                st.write(f"**Qi extraites:** {len(qb)}")
                qc_list=p["qc_list"];judge_map={r["qc_id"]:r for r in p["judge"]}
                matched,orphan=0,0
                for i,q in enumerate(qb):
                    if _is_hdr(q):continue
                    # Try to match against QC triggers
                    best_qc=None;best_score=0
                    for qc in qc_list:
                        if judge_map.get(qc["qc_id"],{}).get("verdict")!="PASS":continue
                        for trig in qc["triggers"]:
                            # Simple keyword overlap
                            ow=set(q.lower().split())&set(trig["text"].lower().split())
                            sc2=len(ow)
                            if sc2>best_score:best_score=sc2;best_qc=qc["qc_id"]
                    if best_qc and best_score>=2:
                        st.write(f"qi_{i+1:02d} | \"{q[:50]}...\" | → **{best_qc}** ✅ (score={best_score})")
                        matched+=1
                    else:
                        st.write(f"qi_{i+1:02d} | \"{q[:50]}...\" | → ❌ ORPHELIN")
                        orphan+=1
                total=matched+orphan
                if total>0:
                    st.divider()
                    st.metric("Taux de couverture",f"{matched}/{total} = {matched/total:.1%}")
                    st.metric("Orphelins",orphan)
            else:st.warning("OCR failed on this PDF.")
        else:st.caption("Upload un sujet pour tester la couverture des QC.")
    # Artifacts tab in sidebar
    with st.sidebar:
        if p and"artifacts"in p:
            st.divider();st.markdown(f"**Artifacts ({len(p['artifacts'])})**")
            for n,m in sorted(p["artifacts"].items()):st.caption(f"{n}: `{m['sha256'][:12]}`")
if __name__=="__main__":main()
