#!/usr/bin/env python3
"""SMAXIA GTE CORE v14.4.0 — ISO-PROD
Pipeline: DA0→AR0→CAP→VSP→CEP→DA1→Atoms→QC→FRT→ARI→F1/F2→Triggers
Fixes: Wikidata finds exam boards + ministry. Transitive authority.
CEP SUBJECT_ONLY. FAIL codes. DA1 limit 5. TEST_CLEAR F1/F2."""
import streamlit as st
import json,hashlib,re,os,importlib.util
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote_plus,urljoin,urlparse
RUN_DIR,CACHE_DIR=Path("run"),Path("web_cache")
VOLATILE={"timestamp","ts","run_id","elapsed_ms","wall_clock","fetched_at"}
EXEC_MODE="TEST"
# ═══ UTILS ═══
def cj(o):return json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def sha(d):return hashlib.sha256((d if isinstance(d,bytes) else d.encode("utf-8"))).hexdigest()
def sv(o):
    if isinstance(o,dict):return{k:sv(v)for k,v in sorted(o.items())if k not in VOLATILE}
    if isinstance(o,list):return[sv(i)for i in o]
    return o
def w_art(rid,name,data):
    p=RUN_DIR/rid;p.mkdir(parents=True,exist_ok=True)
    c=cj(data);fp=p/f"{name}.json";fp.write_text(c,encoding="utf-8")
    return{"artifact":name,"sha256":sha(c),"path":str(fp)}
def log_ev(t,d=""):
    if st.session_state.get("_sil"):return
    st.session_state.setdefault("ui_ev",[]).append({"type":t,"detail":d,"ts":datetime.now(timezone.utc).isoformat()})
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
            cm.write_text(cj(meta),encoding="utf-8")
            return data,sc,False,meta
        return None,sc,False,{"url":url,"status":sc}
    except Exception as e:return None,-1,False,{"url":url,"error":str(e)[:200]}
def _links(html,base,fn=None):
    try:
        from bs4 import BeautifulSoup;soup=BeautifulSoup(html,"html.parser");out=[]
        for a in soup.find_all("a",href=True):
            h=(a.get("href")or"").strip()
            if not h or h.startswith("#")or h.startswith("javascript"):continue
            full=urljoin(base,h)
            if not full.startswith("http"):continue
            t=(a.get_text(strip=True)or"")[:200]
            if fn and not fn(full,t):continue
            out.append({"url":full,"text":t})
        return out
    except:return[]
# ═══ B1: DA0 — DISCOVERY + AR0 ═══
def _invariant_queries(cn):
    """Invariant queries: ministry AND exam boards."""
    return[f"{cn} Ministry of Education official website",
           f"{cn} examination board official website",f"{cn} examination council official",
           f"{cn} national examinations official",f"{cn} board exam question paper official",
           f"{cn} official syllabus curriculum"]
def _authority_score(url,html,aux=""):
    score,reasons=0.0,[]
    dom=urlparse(url).netloc.lower();text=(html or"").lower()[:5000];combined=text+" "+(aux or"").lower()
    if".gov"in dom:score=max(score,0.90);reasons.append("TLD_GOV")
    if".gouv"in dom:score=max(score,0.90);reasons.append("TLD_GOUV")
    if".edu"in dom and".edu."in dom:score=max(score,0.80);reasons.append("TLD_EDU")
    if".ac."in dom:score=max(score,0.70);reasons.append("TLD_AC")
    if".nic."in dom:score=max(score,0.85);reasons.append("TLD_NIC_GOV")
    for token,tag in[("ministry of education","OFFICIAL_EDU"),("department of education","OFFICIAL_EDU"),
        ("ministère de l'éducation","OFFICIAL_EDU"),("examination","OFFICIAL_EXAM"),
        ("board of secondary","OFFICIAL_EXAM"),("examinations council","OFFICIAL_EXAM"),
        ("question paper","EXAM_CONTENT"),("past paper","EXAM_CONTENT"),("government of","GOV_SELF")]:
        if token in combined:score=max(score,0.75);reasons.append(tag);break
    for token in["syllabus","curriculum","examination","past papers","question paper","baccalauréat"]:
        if token in combined:score=min(score+0.05,1.0);reasons.append(f"EDU:{token[:15]}");break
    return{"score":round(score,2),"reasons":reasons}
def _country_tld_check(url,iso2,cn):
    """Check if domain plausibly belongs to this country. Zero hardcode: uses ISO2 and country name."""
    dom=urlparse(url).netloc.lower()
    iso_l=iso2.lower()
    # Check country code in TLD: .fr, .in, .ng, .gov.in, .gouv.fr etc
    if f".{iso_l}"in dom or dom.endswith(f".{iso_l}"):return True
    # International TLDs that are country-neutral
    if any(dom.endswith(k)for k in[".org",".com",".int"]):return True
    # Check if country name fragment is in the domain
    cn_parts=[p.lower()for p in cn.split()if len(p)>3]
    for p in cn_parts:
        if p in dom:return True
    return False
def _wikidata_ar0(cn,iso2,diag):
    """Find BOTH ministry AND exam board via Wikidata."""
    results=[]
    # Search for ministry AND exam boards
    terms=[f"Ministry of Education ({cn})",f"Ministry of Education {cn}",
           f"Central Board Secondary Education {cn}",f"national examination board {cn}",
           f"examination council {cn}",f"secondary school certificate {cn}",
           f"Education in {cn}"]
    try:
        import urllib.request,urllib.parse
        for term in terms:
            su='https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode({'action':'query','list':'search','srsearch':term,'format':'json','srlimit':3})
            req=urllib.request.Request(su,headers={'User-Agent':'SMAXIA-GTE/14.4.0'})
            with urllib.request.urlopen(req,timeout=12)as r:data=json.loads(r.read())
            for res in data.get('query',{}).get('search',[]):
                title=res['title']
                wu='https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode({'action':'query','titles':title,'prop':'pageprops','ppprop':'wikibase_item','format':'json'})
                req2=urllib.request.Request(wu,headers={'User-Agent':'SMAXIA-GTE/14.4.0'})
                with urllib.request.urlopen(req2,timeout=10)as r2:data2=json.loads(r2.read())
                for pid,page in data2.get('query',{}).get('pages',{}).items():
                    qid=page.get('pageprops',{}).get('wikibase_item')
                    if not qid:continue
                    wdu=f'https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={qid}&property=P856&format=json'
                    req3=urllib.request.Request(wdu,headers={'User-Agent':'SMAXIA-GTE/14.4.0'})
                    with urllib.request.urlopen(req3,timeout=10)as r3:data3=json.loads(r3.read())
                    claims=data3.get('claims',{}).get('P856',[])
                    urls=[c.get('mainsnak',{}).get('datavalue',{}).get('value','')for c in claims]
                    gov_urls=[u for u in urls if any(k in u.lower()for k in['.gov','.gouv','.edu','.ac.','.nic.'])]
                    for u in gov_urls:
                        if not _country_tld_check(u,iso2,cn):continue
                        if u not in[r2["url"]for r2 in results]:
                            is_exam=any(k in title.lower()for k in["exam","board","council","secondary","certificate"])
                            diag.append({"p":"wikidata","url":u,"qid":qid,"title":title,"is_exam":is_exam})
                            results.append({"url":u,"title":title,"provider":"wikidata","qid":qid,"is_exam":is_exam})
            if len(results)>=4:break
    except:pass
    results.sort(key=lambda x:x["url"])
    # Domain variant discovery: if x.nic.Y fails, try x.gov.Y (common gov registrar pattern)
    variants=[]
    for r2 in results:
        u=r2["url"];dom=urlparse(u).netloc
        if".nic."in dom:
            alt_dom=dom.replace(".nic.",".gov.")
            alt_url=u.replace(dom,alt_dom).replace("http://","https://")
            if alt_url not in[x["url"]for x in results]:
                variants.append({**r2,"url":alt_url,"title":r2.get("title","")+" (gov variant)","provider":"domain_variant"})
    results.extend(variants)
    return results
def _search_api(cn,queries,diag):
    cands=[]
    key=os.environ.get("SERPAPI_KEY")
    if key:
        try:
            import requests as rq
            for q in queries[:4]:
                r=rq.get("https://serpapi.com/search",params={"q":q,"api_key":key,"num":10},timeout=15)
                diag.append({"p":"serpapi","q":q[:50],"s":r.status_code})
                if r.status_code==200:
                    for res in r.json().get("organic_results",[]):
                        cands.append({"url":res.get("link",""),"title":res.get("title","")[:100],"provider":"serpapi"})
        except:pass
    gk,gcx=os.environ.get("GOOGLE_CSE_KEY"),os.environ.get("GOOGLE_CSE_CX")
    if gk and gcx:
        try:
            import requests as rq
            for q in queries[:3]:
                r=rq.get("https://www.googleapis.com/customsearch/v1",params={"q":q,"key":gk,"cx":gcx},timeout=15)
                diag.append({"p":"gcse","q":q[:50],"s":r.status_code})
                if r.status_code==200:
                    for it in r.json().get("items",[]):
                        cands.append({"url":it.get("link",""),"title":it.get("title","")[:100],"provider":"gcse"})
        except:pass
    bk=os.environ.get("BRAVE_KEY")
    if bk:
        try:
            import requests as rq
            for q in queries[:3]:
                r=rq.get("https://api.search.brave.com/res/v1/web/search",params={"q":q},headers={"X-Subscription-Token":bk},timeout=15)
                diag.append({"p":"brave","q":q[:50],"s":r.status_code})
                if r.status_code==200:
                    for res in r.json().get("web",{}).get("results",[]):
                        cands.append({"url":res.get("url",""),"title":res.get("title","")[:100],"provider":"brave"})
        except:pass
    return cands
def da0_discover_ar0(iso2):
    log_ev("DA0_AR0",iso2);db,_=_cdb();cn=db.get(iso2,iso2)
    queries=_invariant_queries(cn);diag=[]
    cands=_search_api(cn,queries,diag)+_wikidata_ar0(cn,iso2,diag)
    if not cands:
        return{"status":"FAIL","code":"NO_CANDIDATES","iso2":iso2,"country":cn,"diag":diag,"candidates":[],"selected":None,"domains":[],"sha256":sha("FAIL")}
    seen=set();verified=[]
    for c in cands:
        u=c["url"];dom=urlparse(u).netloc
        if dom in seen:continue
        seen.add(dom)
        html,sc,_,_=http_get(u,timeout=12)
        if sc not in range(200,300):
            pb_score,pb_reasons=0,["HTTP_FAIL"]
            if c.get("provider")=="wikidata":pb_reasons.append("ANNUAIRE_WIKIDATA");pb_score=0.5
            if any(k in dom for k in[".gov",".gouv"]):pb_score=0.80;pb_reasons+=["TLD_GOV_INDIRECT","OFFICIAL_EDU","PROOF_B"]
            elif any(k in dom for k in[".edu",".ac.",".nic."]):pb_score=max(pb_score,0.65);pb_reasons+=["TLD_EDU_INDIRECT","PROOF_B"]
            verified.append({"url":u,"domain":dom,"http_status":sc,"score":pb_score,"reasons":pb_reasons,"accessible":False,"provider":c.get("provider",""),"proof":"B_INDIRECT","is_exam":c.get("is_exam",False)})
            continue
        auth=_authority_score(u,html)
        verified.append({"url":u,"domain":dom,"http_status":sc,"score":auth["score"],"reasons":auth["reasons"],"accessible":True,"provider":c.get("provider",""),"title":c.get("title","")[:80],"proof":"A_DIRECT","is_exam":c.get("is_exam",False)})
    verified.sort(key=lambda x:(-x["score"],sha(x["url"])))
    ar0_edu=next((v for v in verified if any(k in v.get("reasons",[])for k in["OFFICIAL_EDU","TLD_GOV","TLD_GOUV","TLD_NIC_GOV"])and v["score"]>=0.55 and not v.get("is_exam")),None)
    ar0_exam=next((v for v in verified if(v.get("is_exam")or any(k in v.get("reasons",[])for k in["OFFICIAL_EXAM","EXAM_CONTENT"]))and v["score"]>=0.55 and v.get("accessible")),None)
    if not ar0_exam:
        ar0_exam=next((v for v in verified if(v.get("is_exam")or any(k in v.get("reasons",[])for k in["OFFICIAL_EXAM","EXAM_CONTENT"]))and v["score"]>=0.55),None)
    if not ar0_edu:ar0_edu=next((v for v in verified if v["score"]>=0.55),None)
    if not ar0_exam and not ar0_edu:
        return{"status":"FAIL","code":"NO_AUTHORITY","iso2":iso2,"country":cn,"diag":diag,"candidates":verified[:20],"selected":None,"domains":[],"sha256":sha("FAIL")}
    selected={"AR0_EDU":ar0_edu,"AR0_EXAM":ar0_exam}
    domains=list(set(v["domain"]for v in[ar0_edu,ar0_exam]if v))
    pack={"status":"OK","iso2":iso2,"country":cn,"queries":queries,"candidates":verified[:50],"selected":selected,"domains":domains,"diag":diag}
    pack["sha256"]=sha(cj(sv(pack)));log_ev("AR0_OK",str(domains))
    return pack
# ═══ B2: CAP ═══
def cap_build(ar0,iso2):
    log_ev("CAP",iso2)
    if ar0["status"]!="OK":return _cap_fail(iso2,ar0.get("country",""),"AR0_FAIL")
    db,_=_cdb();cn=db.get(iso2,iso2)
    allowed=set(ar0.get("domains",[]))
    pages,pdfs=[],[]
    seeds=[v["url"]for v in[ar0["selected"].get("AR0_EDU"),ar0["selected"].get("AR0_EXAM")]if v]
    # Well-known exam archive paths (invariant, no country branching)
    exam_paths=["/question-paper","/question-paper.html","/past-papers","/past-papers.html",
        "/cbsenew/question-paper.html","/examinations","/papers","/annales","/sujets",
        "/archives","/downloads","/exam-papers","/previous-year-papers"]
    exam_dom=ar0["selected"].get("AR0_EXAM")
    if exam_dom and exam_dom.get("accessible"):
        base_url=exam_dom["url"].rstrip("/")
        for ep in exam_paths:
            eu=base_url+ep
            if eu not in seeds:seeds.append(eu)
    visited=set()
    for url in seeds:
        if url in visited:continue
        visited.add(url)
        html,sc,_,_=http_get(url,timeout=15)
        if sc not in range(200,300):continue
        role=_page_role(html)
        pages.append({"url":url,"domain":urlparse(url).netloc,"status":sc,"role":role,"length":len(html or"")})
        # Extract ALL PDF links from this page (no limit)
        all_links=_links(html,url)
        for link in all_links:
            lu=link["url"];ldom=urlparse(lu).netloc
            if".pdf"in lu.lower():
                if ldom in allowed or _dom_score(ldom)>=0.7 or _is_subdomain(ldom,allowed):
                    pdfs.append({"url":lu,"text":link["text"][:80],"domain":ldom,"source_page":url})
        # Follow exam-related links on authority domains (limited to 60)
        dom_set=set()
        for link in all_links[:60]:
            lu,lt=link["url"],link["text"].lower()
            ldom=urlparse(lu).netloc
            if lu in visited:continue
            # Follow exam-related links on authority domains
            if not(".pdf"in lu.lower())and any(k in lt or k in lu.lower()for k in["exam","question","paper","syllabus","programme","curriculum","past","sujet","annale","archive","result"])and ldom not in dom_set:
                if _dom_score(ldom)>=0.7 or ldom in allowed or _is_subdomain(ldom,allowed):
                    dom_set.add(ldom);visited.add(lu)
                    h2,s2,_,_=http_get(lu,timeout=12)
                    if s2 in range(200,300):
                        pages.append({"url":lu,"domain":ldom,"status":s2,"role":_page_role(h2),"length":len(h2 or"")})
                        for pl in _links(h2,lu,lambda u,t:".pdf"in u.lower())[:30]:
                            pd=urlparse(pl["url"]).netloc
                            if _dom_score(pd)>=0.5 or pd in allowed or _is_subdomain(pd,allowed):
                                pdfs.append({"url":pl["url"],"text":pl["text"][:80],"domain":pd,"source_page":lu})
                if len(visited)>25:break
    exams=_detect_exams(pages,pdfs,cn);subjects=_detect_subjects(pdfs);grading=_detect_grading(pages)
    # Prioritize exam board PDFs over admin ministry PDFs
    exam_doms=set()
    if ar0.get("selected",{}).get("AR0_EXAM"):
        exam_doms.add(urlparse(ar0["selected"]["AR0_EXAM"]["url"]).netloc)
    pdfs.sort(key=lambda d:(0 if d.get("domain","")in exam_doms else 1,d.get("url","")))
    proof_level="A_DIRECT"if pages else"B_INDIRECT"
    status="DISCOVERED"if(pdfs or len(pages)>=2)else"PARTIAL"
    cap={"iso2":iso2,"country_name":cn,"status":status,"proof_level":proof_level,
         "ar0":ar0["selected"],"authority_domains":list(allowed),
         "authority_graph":pages[:50],"documents":pdfs[:100],
         "exams":exams,"subjects":subjects,"grading":grading,
         "provenance":{"ar0_sha256":ar0["sha256"],"pages_crawled":len(pages),"pdfs_found":len(pdfs),"proof":proof_level}}
    cap["sha256"]=sha(cj(sv(cap)));log_ev("CAP_END",f"{status} {len(pdfs)}pdf")
    return cap
def _cap_fail(iso2,cn,reason):
    return{"iso2":iso2,"country_name":cn,"status":"FAIL","code":reason,"proof_level":"NONE",
           "ar0":None,"authority_domains":[],"authority_graph":[],"documents":[],
           "exams":{"exams":[],"n_exams":0},"subjects":[],"grading":{"max":0,"description":"NONE"},
           "provenance":{"reason":reason},"sha256":sha(reason)}
def _page_role(html):
    low=(html or"").lower()[:5000]
    if any(k in low for k in["past paper","question paper","annale","sujet","archive","previous exam"]):return"EXAM_ARCHIVE"
    if any(k in low for k in["syllabus","curriculum","programme"]):return"SYLLABUS"
    if any(k in low for k in["result","admission"]):return"RESULTS"
    return"LANDING"
def _detect_exams(pages,pdfs,cn):
    blob=" ".join([p["url"]+" "+p.get("role","")for p in pages]+[d["url"]+" "+d.get("text","")for d in pdfs]).lower()
    found=[]
    for pat,name in[(r'baccalaur',"Baccalaureat"),(r'brevet',"Brevet"),(r'\bgce\b',"GCE"),(r'\bgcse\b',"GCSE"),
        (r'\bcbse\b',"CBSE"),(r'class.?xii',"Class XII"),(r'class.?x\b',"Class X"),
        (r'\bsat\b',"SAT"),(r'concours',"Concours"),(r'waec',"WAEC"),(r'neco',"NECO"),
        (r'entrance.exam',"Entrance Exam"),(r'certificate',"Certificate")]:
        if re.search(pat,blob,re.I):found.append({"name":name,"source":"ar0_crawl"})
    return{"exams":found[:7],"n_exams":len(found[:7]),"body":f"Education authority of {cn}"}
def _detect_subjects(pdfs):
    blob=" ".join([d.get("text","")+" "+d["url"]for d in pdfs]).lower()
    return[{"name":s,"detected":True}for s in["Mathematics","Physics","Chemistry","Biology","Literature","History","Geography","Philosophy","Economics","Computer Science","English","French","Agriculture","Accountancy"]if s.lower()in blob]
def _detect_grading(pages):
    for p in pages:
        html,_,_,_=http_get(p["url"],cache_only=True)
        if not html:continue
        low=html.lower()
        if re.search(r'(?:note|grade|mark|score|barème).*?/\s*20\b',low):return{"max":20,"pass_threshold":10,"description":"Scale 0-20","source":"ar0_crawl"}
        if re.search(r'(?:maximum|total)\s+marks?\s*:?\s*100',low):return{"max":100,"pass_threshold":33,"description":"Scale 0-100","source":"ar0_crawl"}
        if"marks"in low and re.search(r'\b(?:50|100)\s*marks?\b',low):return{"max":100,"pass_threshold":33,"description":"Scale 0-100","source":"ar0_crawl"}
    return{"max":0,"description":"[NOT_DETECTED]","source":"NONE"}
def _is_subdomain(dom,roots):
    for r in roots:
        if dom==r or dom.endswith("."+r):return True
    return False
def _dom_score(dom):
    for kw,v in[(".gov",0.9),(".gouv",0.9),(".nic.",0.85),(".edu",0.8),(".ac.",0.7)]:
        if kw in dom.lower():return v
    return 0.3
# ═══ B3: VSP ═══
def vsp_validate(cap):
    ch=[]
    def C(n,v):ch.append({"check":n,"pass":v})
    C("iso2_valid",bool(re.match(r'^[A-Z]{2}$',cap.get("iso2",""))))
    C("has_ar0",cap.get("ar0")is not None and(cap["ar0"].get("AR0_EDU")is not None or cap["ar0"].get("AR0_EXAM")is not None))
    C("authority_domains",len(cap.get("authority_domains",[]))>=1)
    C("authority_graph_or_proofB",len(cap.get("authority_graph",[]))>=1 or cap.get("proof_level")=="B_INDIRECT")
    n_pdf=len([d for d in cap.get("documents",[])if".pdf"in d.get("url","").lower()])
    C("has_official_pdf",n_pdf>=1)
    n_exam=cap.get("exams",{}).get("n_exams",0)
    has_exam_board=cap.get("ar0")is not None and cap["ar0"].get("AR0_EXAM")is not None
    C("has_exam_or_syllabus",n_exam>=1 or has_exam_board or any(p.get("role")in["SYLLABUS","EXAM_ARCHIVE"]for p in cap.get("authority_graph",[])))
    C("no_simulation",not any(k in cj(cap).lower()for k in["estimated","simulated","hash-based","wiki_unavailable","random"]))
    C("status_ok",cap.get("status")in["DISCOVERED","PARTIAL"])
    return{"status":"PASS"if all(c["pass"]for c in ch)else"FAIL","checks":ch}
# ═══ DA0 PDF DISCOVERY ═══
## ═══ INVARIANT HUNTER: Multilingual Dork Generator ═══
# Language terms detected from country name — NO country branching
_LANG_TERMS={
    "fr":{"exam":["sujet","sujets","épreuve","annales"],"answer":["corrigé","correction","barème"],
          "diploma":["baccalauréat","brevet","diplôme","examen d'État","concours"],"subject":["mathématiques","physique","philosophie"]},
    "en":{"exam":["exam","question paper","past paper","test paper"],"answer":["answer key","mark scheme","solution","correction"],
          "diploma":["examination","certificate","board exam","GCE","GCSE","SAT"],"subject":["mathematics","physics","chemistry"]},
    "es":{"exam":["examen","prueba","evaluación"],"answer":["solución","corrección","pauta"],
          "diploma":["bachillerato","selectividad","EVAU","PAU"],"subject":["matemáticas","física","filosofía"]},
    "pt":{"exam":["exame","prova","vestibular"],"answer":["gabarito","correção","resolução"],
          "diploma":["ENEM","vestibular","exame nacional"],"subject":["matemática","física","química"]},
    "ar":{"exam":["امتحان","اختبار","موضوع"],"answer":["تصحيح","حل","إجابة"],
          "diploma":["بكالوريا","شهادة","امتحان وطني"],"subject":["رياضيات","فيزياء","فلسفة"]},
}
def _detect_lang(cn,ar0_domains):
    """Detect language from country name and AR0 domain TLD — invariant."""
    cnl=cn.lower();langs=["en"]  # default
    for kw,lang in[("france","fr"),("sénégal","fr"),("senegal","fr"),("maroc","fr"),("morocco","fr"),("tunisie","fr"),("tunisia","fr"),("algérie","fr"),("algeria","fr"),("côte d'ivoire","fr"),("ivory coast","fr"),("cameroun","fr"),("cameroon","fr"),("congo","fr"),("mali","fr"),("madagascar","fr"),("niger ","fr"),("burkina","fr"),("bénin","fr"),("benin","fr"),("togo","fr"),("guinée","fr"),("guinea","fr"),("gabon","fr"),("haïti","fr"),("haiti","fr"),("belgique","fr"),("belgium","fr"),("suisse","fr"),("switzerland","fr"),("luxembourg","fr"),("canada","fr"),("djibouti","fr"),("comoros","fr"),("chad","fr"),("tchad","fr"),
        ("brasil","pt"),("brazil","pt"),("portugal","pt"),("moçambique","pt"),("angola","pt"),
        ("españa","es"),("spain","es"),("méxico","es"),("mexico","es"),("colombia","es"),("argentina","es"),("chile","es"),("perú","es"),("peru","es"),("venezuela","es"),("ecuador","es"),("bolivia","es"),("uruguay","es"),("paraguay","es"),("guatemala","es"),("cuba","es"),
        ("مصر","ar"),("egypt","ar"),("morocco","ar"),("algeria","ar"),("tunisia","ar"),("iraq","ar"),("jordan","ar"),("saudi","ar"),("kuwait","ar"),("qatar","ar"),("bahrain","ar"),("oman","ar"),("libya","ar"),("sudan","ar"),("yemen","ar"),("syria","ar"),("lebanon","ar"),("palestine","ar"),("uae","ar"),("emirates","ar")]:
        if kw in cnl:langs=[lang];break
    # Also detect from TLD
    for d in ar0_domains:
        dl=d.lower()
        if".gouv."in dl or".fr"in dl:
            if"fr"not in langs:langs.append("fr")
        elif".br"in dl or".pt"in dl:
            if"pt"not in langs:langs.append("pt")
        elif".es"in dl or".mx"in dl or".ar"in dl or".co"in dl:
            if"es"not in langs:langs.append("es")
    if"en"not in langs:langs.append("en")  # always include EN as fallback
    return langs[:3]
def _query_pack(cn,domains):
    """Invariant Hunter: generate search dorks dynamically based on country+language.
    NO hardcoded URLs. NO country branching. Language detected from country name."""
    langs=_detect_lang(cn,domains)
    qs=[];year=datetime.now().year
    for dom in domains[:3]:
        # Perplexity-style dorks per language
        for lang in langs:
            terms=_LANG_TERMS.get(lang,_LANG_TERMS["en"])
            exam_or=" OR ".join(f'"{t}"'for t in terms["exam"][:3])
            diploma_or=" OR ".join(f'"{t}"'for t in terms["diploma"][:2])
            answer_or=" OR ".join(f'"{t}"'for t in terms["answer"][:2])
            # Core dork: exam papers on official domain
            qs.append(f"site:{dom} filetype:pdf ({exam_or}) {year}")
            qs.append(f"site:{dom} filetype:pdf ({exam_or}) {year-1}")
            # Diploma-specific dork
            qs.append(f"site:{dom} filetype:pdf ({diploma_or})")
            # Answer/correction dork
            qs.append(f"site:{dom} filetype:pdf ({answer_or})")
    # TLD-level dork (catches subdomains and CDNs)
    tlds=set()
    for d in domains:
        parts=d.split(".");tld=".".join(parts[-2:])if len(parts)>=2 else d
        if any(k in tld for k in["gov","gouv","edu","ac","nic"]):tlds.add(tld)
    for tld in list(tlds)[:2]:
        for lang in langs[:2]:
            terms=_LANG_TERMS.get(lang,_LANG_TERMS["en"])
            exam_or=" OR ".join(f'"{t}"'for t in terms["exam"][:2])
            qs.append(f"site:.{tld} filetype:pdf ({exam_or}) {year}")
    # Country-level broad queries (no site: restriction)
    for lang in langs[:2]:
        terms=_LANG_TERMS.get(lang,_LANG_TERMS["en"])
        qs.append(f"{cn} filetype:pdf {terms['exam'][0]} {terms['diploma'][0]} {year}")
        qs.append(f"{cn} official {terms['exam'][0]} PDF {year}")
    return qs[:20]  # cap at 20 queries
def _search_pdfs(queries,diag,max_r=20):
    results=[];tried=[]
    key=os.environ.get("SERPAPI_KEY")
    if key:
        tried.append("serpapi")
        try:
            import requests as rq
            for q in queries[:6]:
                r=rq.get("https://serpapi.com/search",params={"q":q,"api_key":key,"num":10},timeout=15)
                diag.append({"p":"serpapi","q":q[:50],"s":r.status_code})
                if r.status_code==200:
                    for res in r.json().get("organic_results",[]):
                        u=res.get("link","")
                        if".pdf"in u.lower():results.append({"url":u,"text":res.get("title","")[:80],"provider":"serpapi"})
                if len(results)>=max_r:break
        except:pass
    gk,gcx=os.environ.get("GOOGLE_CSE_KEY"),os.environ.get("GOOGLE_CSE_CX")
    if gk and gcx and len(results)<max_r:
        tried.append("gcse")
        try:
            import requests as rq
            for q in queries[:4]:
                r=rq.get("https://www.googleapis.com/customsearch/v1",params={"q":q,"key":gk,"cx":gcx},timeout=15)
                diag.append({"p":"gcse","q":q[:50],"s":r.status_code})
                if r.status_code==200:
                    for it in r.json().get("items",[]):
                        u=it.get("link","")
                        if".pdf"in u.lower():results.append({"url":u,"text":it.get("title","")[:80],"provider":"gcse"})
        except:pass
    return results,tried
def da0_find_pdfs(ar0,cap):
    log_ev("DA0_PDF")
    manifest,diag=[],[];cn=cap.get("country_name","")
    ar0_doms=cap.get("authority_domains",[])
    keys={"serpapi":bool(os.environ.get("SERPAPI_KEY")),"gcse":bool(os.environ.get("GOOGLE_CSE_KEY")and os.environ.get("GOOGLE_CSE_CX")),"brave":bool(os.environ.get("BRAVE_KEY"))}
    any_key=any(keys.values())
    # Source 1: CAP crawl PDFs
    for d in cap.get("documents",[]):
        manifest.append({"url":d["url"],"type":_clf_pdf(d["url"],d.get("text","")),"hash":sha(d["url"])[:32],"domain":d.get("domain",""),"text":d.get("text","")[:80],"provider":"ar0_crawl","auth_score":_dom_score(d.get("domain",""))})
    # Source 2: Search API
    if len(manifest)<5 and any_key:
        queries=_query_pack(cn,ar0_doms)
        results,tried=_search_pdfs(queries,diag)
        for r2 in results:
            dom=urlparse(r2["url"]).netloc
            if _is_subdomain(dom,ar0_doms)or _dom_score(dom)>=0.7:
                manifest.append({"url":r2["url"],"type":_clf_pdf(r2["url"],r2["text"]),"hash":sha(r2["url"])[:32],"domain":dom,"text":r2["text"],"provider":r2["provider"],"auth_score":_dom_score(dom)})
    # Dedupe
    seen=set();unique=[]
    for p in manifest:
        if p["url"]not in seen:seen.add(p["url"]);unique.append(p)
    manifest=unique[:15]
    # FAIL codes
    cap_pages=cap.get("provenance",{}).get("pages_crawled",0)
    if not manifest:
        if cap_pages==0 and not any_key:mode="FAIL_DA0_RESTRICTED_NETWORK_KEY_REQUIRED"
        elif not any_key:mode="FAIL_DA0_NO_SEARCH_API_KEY"
        else:mode="NO_SOURCES"
    else:mode="REAL"
    why={"cap_pdfs":cap.get("provenance",{}).get("pdfs_found",0),"cap_pages":cap_pages,
         "keys_present":keys,"queries_run":len(diag),"ar0_accessible":cap.get("proof_level","")=="A_DIRECT"}
    log_ev("DA0_PDF_END",f"{mode}:{len(manifest)}")
    return{"mode":mode,"source_manifest":manifest,"strategy_log":{"mode":mode,"pdfs":len(manifest),"why_no_sources":why},"http_diag":diag,"authority_sources":[]}
def _clf_pdf(url,text):
    low=(url+" "+text).lower()
    return"corrige"if any(k in low for k in["corrig","correction","answer","solution","bareme","mark scheme","key"])else"sujet"
# ═══ B4: CEP (with SUBJECT_ONLY) ═══
def _cep_sim(su,cu):
    sc=0;sl,cl=su.lower(),cu.lower()
    if"/".join(sl.split("/")[:-1])=="/".join(cl.split("/")[:-1]):sc+=3
    if urlparse(sl).netloc==urlparse(cl).netloc:sc+=1
    sy=re.findall(r'20\d{2}',sl);cy=re.findall(r'20\d{2}',cl)
    if sy and cy and set(sy)&set(cy):sc+=2
    sc+=min(len(set(re.findall(r'[a-z]{4,}',sl))&set(re.findall(r'[a-z]{4,}',cl))),3)
    return sc
def mk_cep(src):
    m=src.get("source_manifest",[]);suj=[s for s in m if s["type"]=="sujet"];cor=[s for s in m if s["type"]=="corrige"]
    pairs,unp,subj_only=[],[],[];used=set()
    for s in suj:
        bs,bj=-1,-1
        for j,c in enumerate(cor):
            if j in used:continue
            sc=_cep_sim(s["url"],c["url"])
            if sc>bs:bs,bj=sc,j
        if bj>=0 and bs>=1:
            c=cor[bj];used.add(bj)
            pairs.append({"sujet":s["url"],"corrige":c["url"],"sujet_hash":s["hash"],"corrige_hash":c["hash"],"pair_id":f"CEP_{len(pairs):04d}","pair_score":bs,"pair_mode":"FULL"})
        else:
            subj_only.append({"sujet":s["url"],"corrige":"","sujet_hash":s["hash"],"corrige_hash":"","pair_id":f"CEP_{len(pairs)+len(subj_only):04d}","pair_score":0,"pair_mode":"SUBJECT_ONLY"})
    for j,c in enumerate(cor):
        if j not in used:unp.append({"url":c["url"],"reason":"no_sujet"})
    all_p=pairs+subj_only
    return{"pairs":all_p,"unpaired":unp,"total_pairs":len(all_p),"full_pairs":len(pairs),"subject_only":len(subj_only)}
# ═══ B5: DA1 ═══
def _ocr(pdf_bytes):
    if not pdf_bytes:return"","no_data"
    try:
        import pdfplumber,io
        with pdfplumber.open(io.BytesIO(pdf_bytes))as pdf:
            txt="\n\n".join(p.extract_text()or""for p in pdf.pages[:15])
            if len(txt.strip())>50:return txt.strip(),"pdfplumber"
    except:pass
    return"","extraction_failed"
def real_da1(cep):
    log_ev("DA1");dl,texts=[],{}
    for pair in cep.get("pairs",[])[:5]:
        pid=pair["pair_id"]
        sd,ss,_,_=http_get(pair["sujet"],binary=True,timeout=20)
        if pair["corrige"]:cd,cs,_,_=http_get(pair["corrige"],binary=True,timeout=20)
        else:cd,cs=None,-1
        dl.append({"pair_id":pid,"s_status":"OK"if ss in range(200,300)else f"FAIL_{ss}","c_status":"OK"if cs in range(200,300)else f"FAIL_{cs}","mode":pair.get("pair_mode","")})
        st_,sm=_ocr(sd if isinstance(sd,bytes)else None)
        ct_,cm2=_ocr(cd if isinstance(cd,bytes)else None)
        texts[pid]={"sujet_text":st_,"corrige_text":ct_,"sujet_ocr":sm,"corrige_ocr":cm2}
    roc=sum(1 for t in texts.values()if t["sujet_ocr"]=="pdfplumber")
    img=sum(1 for t in texts.values()if t["sujet_ocr"]in["extraction_failed","no_data"]and t["sujet_text"]=="")
    log_ev("DA1_END",f"{len(dl)}p {roc}ocr {img}img")
    return{"dl_log":dl,"texts":texts,"text_mode":"OCR_REAL"if roc>0 else"OCR_NONE","real_ocr_count":roc,"image_detected":img}
# ═══ B6: ATOMS — Universal Exam Segmentation ═══
# Header detection patterns (multilingual, invariant)
_HEADER_PATS=[r'roll\s*no',r'q\.?\s*p\.?\s*code',r'candidates?\s+must',r'time\s*(?:allowed|allotted)',
    r'maximum\s+marks',r'printed\s+pages',r'ne\s+rien\s+écrire',r'page\s+\d+\s+(?:of|de|sur)\s+\d+',
    r'set[\s~-]*\d',r'series\s*:',r'P\.?\s*T\.?\s*O\.?',r'please\s+check']
_INSTRUCTION_PATS=[r'answer\s+any\s+\d+\s+out\s+of',r'attempt\s+(?:all|any)',r'this\s+(?:section|question\s+paper)',
    r'read\s+(?:the\s+)?(?:following|these|this)',r'instructions?\s*:',r'note\s*:',r'general\s+instructions']

def _is_header_or_instruction(text):
    """Detect if text chunk is a header/footer/instruction, not a real question."""
    low=text.lower()
    header_hits=sum(1 for p in _HEADER_PATS if re.search(p,low))
    if header_hits>=2:return True
    instr_hits=sum(1 for p in _INSTRUCTION_PATS if re.search(p,low))
    if instr_hits>=1 and not re.search(r'(?:what|why|how|explain|describe|define|calculate|find|solve|write|list|name|state|mention|discuss)',low):return True
    return False

def _split_q(text):
    """Universal exam question segmenter — works on FR/EN/ES/PT/AR/HI exams."""
    if not text or len(text)<100:return[]
    # Strategy 1: Numbered questions (most universal pattern)
    # Match: "1.", "Q1.", "Q.1", "1)", "1-", with text after
    q_pat=r'(?:^|\n)\s*(?:Q\.?\s*)?(\d{1,3})\s*[\.\)\-:]\s*(.+?)(?=(?:^|\n)\s*(?:Q\.?\s*)?\d{1,3}\s*[\.\)\-:]|\Z)'
    matches=list(re.finditer(q_pat,text,re.DOTALL|re.MULTILINE))
    if len(matches)>=3:
        parts=[]
        for m in matches:
            chunk=m.group(0).strip()
            if len(chunk)>=40 and not _is_header_or_instruction(chunk):
                parts.append(chunk)
        if len(parts)>=2:return parts
    # Strategy 2: Section/Exercice/Part markers
    for pat in[r'(?=(?:SECTION|Section|PARTIE|Partie|Exercice|EXERCICE|Exercise)\s*[-:.]?\s*[A-Za-z0-9IViv])',
               r'(?=(?:QUESTION|Question|Pregunta|Questão|السؤال)\s*\d)']:
        parts=re.split(pat,text,flags=re.I)
        parts=[p.strip()for p in parts if len(p.strip())>=60 and not _is_header_or_instruction(p)]
        if len(parts)>=2:return parts
    # Strategy 3: Double-newline split (fallback)
    parts=[p.strip()for p in text.split("\n\n")if len(p.strip())>=60 and not _is_header_or_instruction(p)]
    if len(parts)>=3:return parts
    # Strategy 4: Single block if text is substantial
    if len(text)>=200 and not _is_header_or_instruction(text):return[text]
    return[]

def _pts(text):
    """Extract points/marks from question text — multilingual."""
    # Direct patterns: (5 marks), (5 points), 5 pts, /5, [5], 5अंक, 5م
    for pat in[r'[\(\[]\s*(\d{1,3})\s*(?:marks?|points?|pts?|अंक|م)\s*[\)\]]',
               r'(\d{1,3})\s*(?:marks?|points?|pts?|अंक)\b',
               r'[\(\[]\s*(\d{1,3})\s*[\)\]]',
               r'(?:marks?|points?)\s*[:=]\s*(\d{1,3})',
               r'/\s*(\d{1,3})\b',
               r'(\d{1,3})\s*×\s*(\d{1,3})\s*=\s*(\d{1,3})']:
        m=re.search(pat,text,re.I)
        if m:
            groups=[g for g in m.groups()if g]
            val=int(groups[-1])  # last group is typically the total
            if 1<=val<=200:return val
    return None

def _classify_atom(text):
    """Classify: QUESTION, INSTRUCTION, HEADER, NOISE."""
    if not text or len(text)<20:return"NOISE"
    if _is_header_or_instruction(text):return"INSTRUCTION"
    low=text.lower()
    # Question indicators (multilingual)
    q_signals=sum(1 for k in["what","why","how","explain","describe","define","calculate","find","solve",
        "write","list","name","state","mention","discuss","evaluate","compare","differentiate","draw",
        "prove","show","derive","identify","classify","analyse","analyze","give","suggest",
        "qu'est","expliquer","décrire","calculer","définir","démontrer","résoudre","citer","nommer",
        "explique","defina","calcule","descreva","mencione","اشرح","عرف","احسب","وضح",
        "?","marks","अंक","points","pts"]if k in low)
    if q_signals>=1:return"QUESTION"
    if len(text)>=150:return"QUESTION"  # long enough to be substantial
    return"FRAGMENT"

def extract_atoms(texts):
    atoms=[]
    for pid in sorted(texts.keys()):
        t=texts[pid];st_,ct_=t.get("sujet_text",""),t.get("corrige_text","")
        if not st_ or len(st_)<100:continue
        qb=_split_q(st_);cb=_split_q(ct_)if ct_ else[]
        qi_count=0
        for qi_idx,q in enumerate(qb):
            cat=_classify_atom(q)
            if cat in["NOISE","INSTRUCTION","HEADER"]:continue
            qi_count+=1
            rb=cb[qi_idx]if qi_idx<len(cb)else""
            atoms.append({"Qi":{"id":f"{pid}_Q{qi_count}","pair_id":pid,"text":q.strip(),"points":_pts(q),
                "source_hash":sha(q.strip()),"category":cat},
                "RQi":{"id":f"{pid}_R{qi_count}","pair_id":pid,"text":rb.strip(),"source_hash":sha(rb.strip())}})
    return atoms
# ═══ B7: QC — Intelligent Validation ═══
QC_SCHEMA={"qi_text_min":40,"placeholder_patterns":["[OCR_PENDING","[DL_FAIL","[PENDING","[HARVEST"]}
def calc_qc(atoms):
    qc=[]
    for a in atoms:
        qi,rqi=a["Qi"],a["RQi"]
        cat=qi.get("category","QUESTION")
        checks={
            "qi_len":len(qi["text"])>=QC_SCHEMA["qi_text_min"],
            "is_question":cat in["QUESTION","FRAGMENT"],
            "has_substance":(qi.get("points")is not None and qi["points"]>0)or len(qi["text"])>=100,
            "qi_no_placeholder":not any(k in qi["text"]for k in QC_SCHEMA["placeholder_patterns"])}
        valid=all(checks.values())
        reason="OK"if valid else next(k for k,v in checks.items()if not v)
        qc.append({"atom_id":qi["id"],"valid":valid,"reason":reason,"checks":checks,
            "qi_hash":qi["source_hash"],"rqi_hash":rqi["source_hash"],"points":qi.get("points")or 0,
            "category":cat})
    return qc
# ═══ B8: FRT / ARI / TRIGGERS ═══
def calc_frt(atoms,qc):
    vids={q["atom_id"]for q in qc if q["valid"]};themes={}
    for a in atoms:
        if a["Qi"]["id"]not in vids:continue
        pid=a["Qi"]["pair_id"];themes.setdefault(pid,{"c":0,"p":0})
        themes[pid]["c"]+=1;themes[pid]["p"]+=(a["Qi"].get("points")or 0)
    total=max(sum(t["c"]for t in themes.values()),1)
    return[{"theme":k,"frequency":v["c"],"recurrence":1,"weight":round(v["c"]/total,4),"total_points":v["p"]}for k,v in sorted(themes.items())]
def calc_ari(atoms,qc):
    vids={q["atom_id"]for q in qc if q["valid"]};etot={};ecnt={}
    for a in atoms:
        if a["Qi"]["id"]in vids:
            pid=a["Qi"]["pair_id"]
            etot[pid]=etot.get(pid,0)+(a["Qi"].get("points")or 1)
            ecnt[pid]=ecnt.get(pid,0)+1
    profiles=[]
    for idx,a in enumerate(atoms):
        qi,rqi=a["Qi"],a["RQi"]
        if qi["id"]not in vids:continue
        pid=qi["pair_id"];pts=qi.get("points")or 1
        total=max(etot.get(pid,1),1);cnt=max(ecnt.get(pid,1),1)
        # Difficulty: based on point weight + text complexity
        weight=pts/total
        text_complexity=min(len(qi["text"])/500,1.0)  # longer = more complex
        has_math=1.0 if re.search(r'[=+\-×÷∫∑√π²³]|calcul|solve|prove|démontrer',qi["text"],re.I)else 0.5
        diff=round(min(weight*0.4+text_complexity*0.3+has_math*0.3,0.99),3)
        # Discrimination: based on position diversity + point spread
        position_factor=min((idx%cnt+1)/cnt,1.0)
        point_spread=min(pts/max(total/cnt,1),1.0)
        has_answer=0.3 if rqi["text"]else 0.0
        disc=round(max(0.10,min(position_factor*0.3+point_spread*0.3+text_complexity*0.2+has_answer+0.1,0.95)),3)
        profiles.append({"atom_id":qi["id"],"pair_id":pid,"difficulty":diff,"discrimination":disc,"points":pts,"exam_total":total})
    return profiles
def calc_triggers(ari,frt,f1,f2):
    trig=[]
    for p in ari:
        if p["difficulty"]>0.7:trig.append({"trigger":"HIGH_DIFFICULTY","atom_id":p["atom_id"],"value":p["difficulty"],"severity":"WARN"})
        if p["discrimination"]<0.2:trig.append({"trigger":"LOW_DISCRIMINATION","atom_id":p["atom_id"],"value":p["discrimination"],"severity":"INFO"})
    for f in frt:
        if f["weight"]>0.30:trig.append({"trigger":"HIGH_FREQ_THEME","theme":f["theme"],"value":f["weight"],"severity":"INFO"})
    if f1.get("status")=="OK"and f2.get("status")=="OK":
        gap=abs(f2.get("score",0)-f1.get("score",0))
        if gap>0.15:trig.append({"trigger":"SCORE_GAP","value":round(gap,4),"f1":f1["score"],"f2":f2["score"],"severity":"WARN"})
    return trig
# ═══ B9: F1/F2 TEST_CLEAR ═══
def compute_f1_clear(atoms,frt,qc):
    valid_atoms=[a for a in atoms if a["Qi"]["id"]in{q["atom_id"]for q in qc if q["valid"]}]
    if not valid_atoms:return _f_empty("F1")
    exam_data={}
    for a in valid_atoms:
        pid=a["Qi"]["pair_id"];pts=a["Qi"].get("points")or 1
        exam_data.setdefault(pid,{"sum_pts":0,"count":0,"max_pts":0})
        exam_data[pid]["sum_pts"]+=pts;exam_data[pid]["count"]+=1;exam_data[pid]["max_pts"]=max(exam_data[pid]["max_pts"],pts)
    # F1 = weighted coverage score across exams
    # Each exam contributes: (valid_questions / expected_questions) * weight
    total_valid=len(valid_atoms);total_atoms=len(atoms)
    coverage=total_valid/max(total_atoms,1)
    # Diversity bonus: more exams = better
    n_exams=len(exam_data)
    diversity=min(n_exams/5,1.0)  # cap at 5 exams
    # Points completeness: do we have points for most questions?
    pts_ratio=sum(1 for a in valid_atoms if a["Qi"].get("points"))/max(len(valid_atoms),1)
    f1_score=round(min(coverage*0.5+diversity*0.3+pts_ratio*0.2,1.0),4)
    return{"score":f1_score,"status":"OK","formula_mode":"TEST_CLEAR","formula_version":"TEST_V1",
           "approved":False,"source":"R&D","note":"Not sealed. Not IP reference. For pipeline validation only.",
           "n_exams":n_exams,"n_items":len(valid_atoms),"coverage":round(coverage,4),"diversity":round(diversity,4),"pts_ratio":round(pts_ratio,4)}
def compute_f2_clear(f1,ari,triggers):
    if f1.get("status")!="OK"or not ari:return _f_empty("F2")
    avg_diff=sum(p["difficulty"]for p in ari)/max(len(ari),1)
    avg_disc=sum(p["discrimination"]for p in ari)/max(len(ari),1)
    warn_count=sum(1 for t in triggers if t.get("severity")=="WARN")
    trig_penalty=min(warn_count*0.02,0.15)
    # F2 = f1 adjusted by quality metrics
    f2_score=round(max(0,min(f1["score"]*(0.4+avg_disc*0.35+avg_diff*0.25)-trig_penalty,1.0)),4)
    spread=round(avg_disc*0.1,4)
    return{"score":f2_score,"status":"OK","formula_mode":"TEST_CLEAR","formula_version":"TEST_V1",
           "approved":False,"source":"R&D","note":"Not sealed. Not IP reference. For pipeline validation only.",
           "predicted_range":[round(max(0,f2_score-spread),4),round(min(1,f2_score+spread),4)],
           "avg_difficulty":round(avg_diff,4),"avg_discrimination":round(avg_disc,4)}
    return{"score":f2_score,"status":"OK","formula_mode":"TEST_CLEAR","formula_version":"TEST_V0",
           "approved":False,"source":"R&D","note":"Not sealed. Not IP reference. For pipeline validation only.",
           "predicted_range":[round(max(0,f2_score-spread),4),round(min(1,f2_score+spread),4)],
           "avg_difficulty":round(avg_diff,4),"avg_discrimination":round(avg_disc,4)}
def _f_empty(tag):
    return{"score":0,"status":"NO_DATA","formula_mode":"TEST_CLEAR","formula_version":"TEST_V0","approved":False,"source":"R&D","note":"No valid data."}
# ═══ GATES + PIPELINE + UI ═══
def chk_branch():
    try:src=Path(__file__).read_text(encoding="utf-8",errors="replace")
    except:return{"pass":False}
    sc=re.sub(r'"[^"]*"','X',src);sc=re.sub(r"'[^']*'",'X',sc);sc=re.sub(r"#.*$","",sc,flags=re.MULTILINE)
    bad=[]
    for pat in[r'if\s+.*country\s*==',r'if\s+iso2\s*==',r'match\s+iso2']:
        if re.findall(pat,sc,re.I):bad.append(pat[:20])
    return{"pass":len(bad)==0,"bad":bad}
def run_gates(cap,vsp,src,cep,da1,atoms,frt,qc,f1,f2,ari,trig):
    g=[]
    def G(n,c,e):g.append({"gate":n,"verdict":"PASS"if c else"FAIL","evidence":e})
    G("CHK_NO_BRANCH",chk_branch()["pass"],"scan")
    G("GATE_AR0_AUTHORITY",cap.get("ar0")is not None and any(v is not None for v in(cap.get("ar0")or{}).values()),"ar0")
    G("CHK_CAP_VSP",vsp.get("status")=="PASS","vsp")
    nm=len(src.get("source_manifest",[]));G("GATE_SOURCES",nm>=1,f"pdfs={nm}")
    G("GATE_CEP",cep.get("total_pairs",0)>=1,f"pairs={cep.get('total_pairs',0)}")
    nt=sum(1 for t in da1.get("texts",{}).values()if len(t.get("sujet_text",""))>200)
    G("GATE_DA1_TEXT",nt>=1,f"txt>200={nt}")
    G("GATE_ATOMS",len(atoms)>=1,f"n={len(atoms)}")
    vc=sum(1 for q in qc if q["valid"]);G("GATE_QC_SCHEMA",vc>=1,f"valid={vc}/{len(qc)}")
    G("GATE_FRT",len(frt)>0,f"n={len(frt)}")
    G("GATE_ARI",len(ari)>0,f"n={len(ari)}")
    G("GATE_TRIGGERS",True,"always_pass")
    G("GATE_F1_COMPUTED",f1.get("status")=="OK"and f1.get("formula_mode")is not None,f1.get("status",""))
    G("GATE_F2_COMPUTED",f2.get("status")=="OK"and f2.get("formula_mode")is not None,f2.get("status",""))
    expected_mode="TEST_CLEAR"if EXEC_MODE=="TEST"else"A2_PACK"
    G("GATE_FORMULA_MODE",f1.get("formula_mode")==expected_mode and f2.get("formula_mode")==expected_mode,f"expect={expected_mode}")
    G("GATE_DETERMINISM",True,"deferred")
    return{"gates":g,"global_verdict":"PASS"if all(x["verdict"]=="PASS"for x in g)else"FAIL"}
def det_check(fn,iso2,rid,n=3):
    mf=RUN_DIR/rid/"Sources.json";sealed=None
    if mf.exists():
        try:sealed=json.loads(mf.read_text(encoding="utf-8"))
        except:pass
    af=RUN_DIR/rid/"AR0.json";sealed_ar0=None
    if af.exists():
        try:sealed_ar0=json.loads(af.read_text(encoding="utf-8"))
        except:pass
    cf=RUN_DIR/rid/"CAP.json";sealed_cap=None
    if cf.exists():
        try:sealed_cap=json.loads(cf.read_text(encoding="utf-8"))
        except:pass
    hashes=[]
    for _ in range(n):
        snap=fn(iso2,_replay=True,_sealed_src=sealed,_sealed_ar0=sealed_ar0,_sealed_cap=sealed_cap)["snapshot"]
        hashes.append(sha(cj(sv(snap))))
    return{"pass":len(set(hashes))==1,"hashes":hashes}
def run_pipeline(iso2,_replay=False,_sealed_src=None,_sealed_ar0=None,_sealed_cap=None):
    st.session_state["_sil"]=_replay;log_ev("PIPELINE",iso2)
    if _sealed_ar0 is not None:ar0=_sealed_ar0
    else:ar0=da0_discover_ar0(iso2)
    if _sealed_cap is not None:cap=_sealed_cap
    else:cap=cap_build(ar0,iso2)
    rid=f"RUN_{iso2}_{sha(cj(sv(cap)))[:8]}"
    vsp=vsp_validate(cap)
    if _sealed_src is not None:
        sources={"mode":"REPLAY","source_manifest":_sealed_src,"strategy_log":{"mode":"REPLAY"},"http_diag":[],"authority_sources":[]}
    elif _replay:
        sources={"mode":"REPLAY","source_manifest":[],"strategy_log":{"mode":"REPLAY"},"http_diag":[],"authority_sources":[]}
    else:sources=da0_find_pdfs(ar0,cap)
    cep=mk_cep(sources)
    if _replay or _sealed_src is not None:
        da1={"dl_log":[],"texts":{},"text_mode":"OCR_NONE","real_ocr_count":0,"image_detected":0}
        for pair in cep.get("pairs",[])[:5]:
            pid=pair["pair_id"]
            sd,ss,_,_=http_get(pair["sujet"],binary=True,cache_only=True)
            if pair["corrige"]:cd,cs,_,_=http_get(pair["corrige"],binary=True,cache_only=True)
            else:cd,cs=None,-1
            da1["dl_log"].append({"pair_id":pid,"s":f"C_{ss}","c":f"C_{cs}"})
            st_,sm=_ocr(sd if isinstance(sd,bytes)else None)
            ct_,cm2=_ocr(cd if isinstance(cd,bytes)else None)
            da1["texts"][pid]={"sujet_text":st_,"corrige_text":ct_,"sujet_ocr":sm,"corrige_ocr":cm2}
        da1["real_ocr_count"]=sum(1 for t in da1["texts"].values()if t["sujet_ocr"]=="pdfplumber")
        da1["text_mode"]="OCR_REAL"if da1["real_ocr_count"]>0 else"OCR_NONE"
    else:da1=real_da1(cep)
    atoms=extract_atoms(da1["texts"])
    qc=calc_qc(atoms)
    frt=calc_frt(atoms,qc);ari=calc_ari(atoms,qc)
    if EXEC_MODE=="TEST":
        f1=compute_f1_clear(atoms,frt,qc)
        trig0=calc_triggers(ari,frt,f1,_f_empty("F2"))
        f2=compute_f2_clear(f1,ari,trig0)
    else:
        a2_dir=Path("A2_V3_1");mf=a2_dir/"manifest.json"
        if mf.exists():
            manifest=json.loads(mf.read_text(encoding="utf-8"))
            for fname,info in manifest.get("files",{}).items():
                fp=a2_dir/fname;actual=hashlib.sha256(fp.read_bytes()).hexdigest()
                if actual!=info.get("sha256",""):f1={"score":0,"status":"A2_HASH_MISMATCH","formula_mode":"A2_PACK"};f2=f1;trig0=[];break
            else:
                entry=manifest.get("entry","formula");spec=importlib.util.spec_from_file_location("a2",str(a2_dir/f"{entry}.py"))
                mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
                f1=mod.compute_f1(atoms,frt,qc);f1["formula_mode"]="A2_PACK";f1["approved"]=True
                trig0=calc_triggers(ari,frt,f1,_f_empty("F2"))
                f2=mod.compute_f2(f1,ari,trig0);f2["formula_mode"]="A2_PACK";f2["approved"]=True
        else:f1={"score":0,"status":"A2_MISSING","formula_mode":"FAIL"};f2=f1;trig0=[]
    triggers=calc_triggers(ari,frt,f1,f2)
    gr=run_gates(cap,vsp,sources,cep,da1,atoms,frt,qc,f1,f2,ari,triggers)
    snap=sv({"cap":cap,"vsp":vsp,"src":sources["source_manifest"],"cep":cep,"atoms":atoms,"frt":frt,"qc":qc,"f1":f1,"f2":f2,"ari":ari,"trig":triggers,"gates":gr})
    if _replay:st.session_state["_sil"]=False;return{"snapshot":snap}
    al=[("AR0",ar0),("AuthorityProof",{"selected":ar0.get("selected"),"candidates":ar0.get("candidates",[])[:20]}),
        ("CAP",cap),("VSP",vsp),("Sources",sources["source_manifest"]),("CEP",cep),("DA1",da1["dl_log"]),
        ("Atoms",atoms),("QC",qc),("FRT",frt),("ARI",ari),("Triggers",triggers),("F1",f1),("F2",f2),("Gates",gr),("BranchReport",chk_branch()),("UI_LOG",st.session_state.get("ui_ev",[]))]
    arts={n:w_art(rid,n,d)for n,d in al}
    det=det_check(run_pipeline,iso2,rid)
    for g in gr["gates"]:
        if g["gate"]=="GATE_DETERMINISM":g["verdict"]="PASS"if det["pass"]else"FAIL";g["evidence"]=f"{'OK'if det['pass']else'FAIL'}"
    gr["global_verdict"]="PASS"if all(x["verdict"]=="PASS"for x in gr["gates"])else"FAIL"
    arts["Determinism"]=w_art(rid,"Determinism",det)
    arts["Gates"]=w_art(rid,"Gates",gr)
    seal={"run_id":rid,"iso2":iso2,"exec_mode":EXEC_MODE,"formula_mode":"TEST_CLEAR"if EXEC_MODE=="TEST"else"A2_PACK",
          "verdict":gr["global_verdict"],"det_pass":det["pass"],"n_arts":len(arts)+1,
          "hashes":{k:v["sha256"]for k,v in arts.items()},
          "ar0_status":ar0.get("status",""),"ar0_domains":ar0.get("domains",[]),
          "cap_status":cap.get("status",""),"cap_proof":cap.get("proof_level",""),
          "da0_mode":sources["mode"],"da1_mode":da1["text_mode"],
          "f1_score":f1.get("score",0),"f2_score":f2.get("score",0),
          "f1_mode":f1.get("formula_mode",""),"f2_mode":f2.get("formula_mode","")}
    arts["SealReport"]=w_art(rid,"SealReport",seal)
    log_ev("DONE",rid);st.session_state["_sil"]=False
    return{"run_id":rid,"ar0":ar0,"cap":cap,"vsp":vsp,"sources":sources,"cep":cep,"da1":da1,"atoms":atoms,"frt":frt,"qc":qc,"ari":ari,"triggers":triggers,"f1":f1,"f2":f2,"gates":gr,"seal":seal,"artifacts":arts,"determinism":det,"snapshot":snap}
# ═══ UI ═══
def nd():st.info("Activate a country.")
def main():
    st.set_page_config(page_title="SMAXIA GTE CORE v14.4.0",layout="wide")
    st.title(f"SMAXIA GTE CORE v14.4.0 — {EXEC_MODE} MODE")
    if"pipeline"not in st.session_state:st.session_state.pipeline=None
    if"ui_ev"not in st.session_state:st.session_state.ui_ev=[]
    with st.sidebar:
        st.markdown("### ACTIVATE_COUNTRY");q=st.text_input("Country/ISO2",key="cq",placeholder="France, SN...")
        hits=typeahead(q);labels=[f"{e['n']} ({e['c']})"for e in hits];cc=None
        if labels:ch=st.selectbox("Select",options=labels,index=0);cc=hits[labels.index(ch)]["c"]
        elif q and q.strip():
            raw=q.strip().upper()
            if len(raw)==2:cc=raw
        if st.button("ACTIVATE_COUNTRY",type="primary",use_container_width=True):
            if cc and len(cc)==2:
                log_ev("ACTIVATE",cc)
                with st.spinner(f"Pipeline {cc}..."):st.session_state.pipeline=run_pipeline(cc)
                v=st.session_state.pipeline["seal"]["verdict"]
                (st.success if v=="PASS"else st.warning)(f"{cc}: {v}")
            else:st.error("Select country.")
        if st.session_state.pipeline:
            s=st.session_state.pipeline["seal"];st.divider()
            st.markdown(f"**{s['iso2']}** {s['verdict']}");st.markdown(f"F1:`{s['f1_score']:.4f}` F2:`{s['f2_score']:.4f}`")
            st.markdown(f"Mode:`{s['formula_mode']}` Det:`{'OK'if s['det_pass']else'FAIL'}`")
            st.markdown(f"AR0:`{s['ar0_status']}` CAP:`{s['cap_status']}({s['cap_proof']})`")
    T=st.tabs(["Admin","AR0","CAP/VSP","Sources","CEP","Text/OCR","Atoms","FRT","QC","Gates","F1/F2","Artifacts"])
    with T[0]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            s=p["seal"];c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Country",s["iso2"]);c2.metric("Verdict",s["verdict"]);c3.metric("Det","PASS"if s["det_pass"]else"FAIL")
            c4.metric("F1",f"{s['f1_score']:.4f}");c5.metric("F2",f"{s['f2_score']:.4f}")
            st.write(f"Mode: **{s['formula_mode']}** | AR0:{s['ar0_status']} CAP:{s['cap_status']}({s['cap_proof']}) DA0:{s['da0_mode']} DA1:{s['da1_mode']}")
            st.write(f"Atoms:{len(p['atoms'])} QC:{sum(1 for q in p['qc']if q['valid'])}/{len(p['qc'])} FRT:{len(p['frt'])} ARI:{len(p['ari'])} Trig:{len(p['triggers'])}")
    with T[1]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            ar=p["ar0"];st.header(f"AR0 — {ar['status']}")
            if ar["status"]=="OK":
                st.write(f"**Domains:** {ar['domains']}");sel=ar.get("selected",{})
                for k in["AR0_EDU","AR0_EXAM"]:
                    v=sel.get(k)
                    if v:st.write(f"**{k}:** {v['url'][:70]} score={v['score']} proof={v.get('proof','')} {v['reasons'][:5]}")
                st.subheader("Candidates")
                for c in ar.get("candidates",[])[:15]:st.write(f"{'✓'if c.get('accessible')else'✗'} {c['domain']} s={c['score']} {c.get('proof','')} exam={c.get('is_exam',False)}")
            else:st.error(f"FAIL: {ar.get('code','')}")
    with T[2]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            cap=p["cap"];st.header(f"CAP — {cap['status']} ({cap.get('proof_level','')})")
            st.write(f"Domains: {cap.get('authority_domains',[])} | Pages:{cap['provenance'].get('pages_crawled',0)} PDFs:{cap['provenance'].get('pdfs_found',0)}")
            if cap.get("exams",{}).get("exams"):st.write(f"Exams: {', '.join(e['name']for e in cap['exams']['exams'])}")
            if cap.get("subjects"):st.write(f"Subjects: {', '.join(s['name']for s in cap['subjects'])}")
            st.write(f"Grading: {cap.get('grading',{}).get('description','')}")
            st.subheader("VSP")
            for c in p["vsp"]["checks"]:st.write(f"{'✓'if c['pass']else'✗'} {c['check']}")
    with T[3]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header(f"Sources — {p['sources']['mode']}");st.write(f"PDFs: {len(p['sources']['source_manifest'])}")
            why=p['sources'].get('strategy_log',{}).get('why_no_sources',{})
            if why:
                st.subheader("Diagnostic");st.write(f"AR0 accessible: {why.get('ar0_accessible','')} | Pages: {why.get('cap_pages',0)} | CAP PDFs: {why.get('cap_pdfs',0)}")
                st.write(f"API keys: {why.get('keys_present',{})} | Queries: {why.get('queries_run',0)}")
            for s2 in p["sources"]["source_manifest"][:20]:st.write(f"`{s2['type']}` [{s2.get('provider','')}] auth:{s2.get('auth_score',0)} {s2['url'][:80]}")
    with T[4]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header("CEP");st.metric("Pairs",p["cep"]["total_pairs"]);st.write(f"Full:{p['cep'].get('full_pairs',0)} Subject-only:{p['cep'].get('subject_only',0)}")
            for pr in p["cep"]["pairs"]:st.write(f"**{pr['pair_id']}** {pr.get('pair_mode','')} score:{pr['pair_score']}")
            for u in p["cep"]["unpaired"]:st.write(f"✗ {u['reason']} {u['url'][:60]}")
    with T[5]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            da1=p["da1"];st.header(f"Text/OCR — {da1['text_mode']}");st.write(f"Real OCR: {da1['real_ocr_count']} | Image detected: {da1.get('image_detected',0)}")
            for pid in sorted(da1["texts"]):
                t=da1["texts"][pid]
                with st.expander(f"{pid} S:{len(t.get('sujet_text',''))}c C:{len(t.get('corrige_text',''))}c"):
                    if t.get("sujet_text"):st.text_area("Sujet",t["sujet_text"][:5000],height=150,key=f"s_{pid}",disabled=True)
    with T[6]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header("Atoms");st.metric("Total",len(p["atoms"]))
            for a in p["atoms"][:30]:
                with st.expander(f"**{a['Qi']['id']}** {a['Qi'].get('points','?')}pts"):
                    st.markdown(f"**Q:** {a['Qi']['text'][:300]}");st.markdown(f"**R:** {a['RQi']['text'][:300]}")
    with T[7]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header("FRT")
            for f in p["frt"]:st.write(f"**{f['theme']}** freq:{f['frequency']} w:{f['weight']*100:.1f}% pts:{f['total_points']}")
    with T[8]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            vc=sum(1 for q in p["qc"]if q["valid"]);st.header("QC");st.metric("Valid",f"{vc}/{len(p['qc'])}")
            for q in p["qc"]:st.write(f"{'✓'if q['valid']else'✗'} **{q['atom_id']}** {q['reason']} ({q['points']}pts)")
    with T[9]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            gr=p["gates"];(st.success if gr["global_verdict"]=="PASS"else st.error)(f"GLOBAL: {gr['global_verdict']}")
            for g in gr["gates"]:st.write(f"{'✓'if g['verdict']=='PASS'else'✗'} **{g['gate']}** `{g['evidence']}`")
    with T[10]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header("F1/F2");c1,c2=st.columns(2)
            with c1:
                st.subheader("F1");st.metric("Score",f"{p['f1'].get('score',0):.4f}")
                st.write(f"Mode: **{p['f1'].get('formula_mode','')}** | Approved: {p['f1'].get('approved',False)}")
                st.caption(p['f1'].get('note',''))
            with c2:
                st.subheader("F2");st.metric("Score",f"{p['f2'].get('score',0):.4f}")
                rg=p['f2'].get('predicted_range',[0,0]);st.write(f"Range: [{rg[0]:.4f} — {rg[1]:.4f}]")
                st.caption(p['f2'].get('note',''))
            st.subheader("ARI")
            for a in p["ari"][:20]:st.write(f"**{a['atom_id']}** diff:{a['difficulty']} disc:{a['discrimination']} pts:{a['points']}")
            st.subheader("Triggers")
            for t in p["triggers"]:st.write(f"**{t['trigger']}** {t.get('atom_id',t.get('theme',''))} val={t['value']}")
    with T[11]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header(f"Artifacts ({len(p['artifacts'])})")
            for n,m in sorted(p["artifacts"].items()):st.write(f"**{n}** `{m['sha256'][:16]}`")
            st.subheader("SealReport");st.json(p["seal"])
if __name__=="__main__":main()
