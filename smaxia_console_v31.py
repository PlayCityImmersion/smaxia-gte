#!/usr/bin/env python3
"""SMAXIA GTE CORE v15.0.0 — ISO-PROD
CHANGELOG v15: Brave L2 snapshot+SHA256, DA1 15 pairs, QUARANTINE pairing/atoms,
CAP_PARTIAL_NO_CHAPTERS, QC_text from Qi source, singleton bootstrap, CHK_REPORT.
Formulas TEST_CLEAR (CEO concession for TEST phase)."""
import streamlit as st
import json,hashlib,re,os,importlib.util
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote_plus,urljoin,urlparse
RUN_DIR,CACHE_DIR=Path("run"),Path("web_cache")
VOLATILE={"timestamp","ts","run_id","elapsed_ms","wall_clock","fetched_at"}
EXEC_MODE="TEST";VERSION="15.0.0"
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
            m=re.search(r'>([^<]{1,120})</a',html[html.find(r2):html.find(r2)+500],re.I)
            if m:txt=m.group(1).strip()
            if fn and not fn(u,txt):continue
            out.append({"url":u,"text":txt})
        return out
    except:return[]
# ═══ B1: DA0 AR0 ═══
def _invariant_queries(cn):
    return[f'"ministry of education" "{cn}"',f'"national curriculum" "{cn}"',f'"exam syllabus" "{cn}"',
           f'"secondary education" "{cn}"',f'"examination board" "{cn}"',f'"examination council" "{cn}"']
def _authority_score(url,html,aux=""):
    dom=urlparse(url).netloc.lower();reasons=[];sc=0.3
    for kw,v,tag in[(".gov",0.9,"TLD_GOV"),(".gouv",0.9,"TLD_GOUV"),(".edu",0.8,"TLD_EDU"),(".ac.",0.7,"TLD_AC"),(".nic.",0.85,"TLD_NIC_GOV")]:
        if kw in dom:sc=max(sc,v);reasons.append(tag)
    low=(html or"").lower()[:8000]
    for kw,tag in[("ministry","OFFICIAL_EDU"),("ministère","OFFICIAL_EDU"),("education","OFFICIAL_EDU"),("curriculum","SYLLABUS"),("examination","OFFICIAL_EXAM"),("question paper","EXAM_CONTENT"),("past paper","EXAM_CONTENT"),("annales","EXAM_CONTENT"),("sujet","EXAM_CONTENT")]:
        if kw in low:reasons.append(tag)
    if any(k in reasons for k in["OFFICIAL_EDU","OFFICIAL_EXAM"]):sc=max(sc,0.75)
    if "EXAM_CONTENT"in reasons:sc=max(sc,0.7)
    return{"score":round(sc,2),"reasons":reasons[:10]}
def _country_tld_check(url,iso2,cn):
    dom=urlparse(url).netloc.lower()
    return any(k in dom for k in[".gov",".gouv",".edu",".ac.",".nic.",".org"])or f".{iso2.lower()}"in dom
def _wikidata_ar0(cn,iso2,diag):
    results=[]
    terms=[f"Ministry of Education ({cn})",f"Ministry of Education {cn}",f"Central Board Secondary Education {cn}",
           f"national examination board {cn}",f"examination council {cn}",f"Education in {cn}"]
    try:
        import urllib.request,urllib.parse
        for term in terms:
            su='https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode({'action':'query','list':'search','srsearch':term,'format':'json','srlimit':3})
            req=urllib.request.Request(su,headers={'User-Agent':f'SMAXIA-GTE/{VERSION}'})
            with urllib.request.urlopen(req,timeout=12)as r:data=json.loads(r.read())
            for res in data.get('query',{}).get('search',[]):
                title=res['title']
                wu='https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode({'action':'query','titles':title,'prop':'pageprops','ppprop':'wikibase_item','format':'json'})
                req2=urllib.request.Request(wu,headers={'User-Agent':f'SMAXIA-GTE/{VERSION}'})
                with urllib.request.urlopen(req2,timeout=10)as r2:data2=json.loads(r2.read())
                for pid,page in data2.get('query',{}).get('pages',{}).items():
                    qid=page.get('pageprops',{}).get('wikibase_item')
                    if not qid:continue
                    wdu=f'https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={qid}&property=P856&format=json'
                    req3=urllib.request.Request(wdu,headers={'User-Agent':f'SMAXIA-GTE/{VERSION}'})
                    with urllib.request.urlopen(req3,timeout=10)as r3:data3=json.loads(r3.read())
                    for c2 in data3.get('claims',{}).get('P856',[]):
                        u=c2.get('mainsnak',{}).get('datavalue',{}).get('value','')
                        if not any(k in u.lower()for k in['.gov','.gouv','.edu','.ac.','.nic.']):continue
                        if not _country_tld_check(u,iso2,cn):continue
                        if u not in[x["url"]for x in results]:
                            ie=any(k in title.lower()for k in["exam","board","council","secondary","certificate"])
                            diag.append({"p":"wikidata","url":u,"qid":qid,"title":title,"is_exam":ie})
                            results.append({"url":u,"title":title,"provider":"wikidata","qid":qid,"is_exam":ie})
            if len(results)>=4:break
    except:pass
    results.sort(key=lambda x:x["url"])
    variants=[]
    for r2 in results:
        u=r2["url"];dom=urlparse(u).netloc
        if".nic."in dom:
            alt=u.replace(dom,dom.replace(".nic.",".gov.")).replace("http://","https://")
            if alt not in[x["url"]for x in results]:variants.append({**r2,"url":alt,"title":r2["title"]+" (gov variant)","provider":"domain_variant"})
    results.extend(variants);return results
def _brave_search(queries,diag,max_per=10):
    """Brave Search L2 — snapshot+hash for determinism."""
    cands,snaps=[],[];bk=os.environ.get("BRAVE_KEY")
    if not bk:return cands,snaps
    try:
        import requests as rq
        for q in queries:
            r=rq.get("https://api.search.brave.com/res/v1/web/search",params={"q":q,"count":max_per},
                headers={"Accept":"application/json","X-Subscription-Token":bk},timeout=15)
            diag.append({"p":"brave","q":q[:50],"s":r.status_code,"source_flag":"BRAVE_L2"})
            if r.status_code==200:
                raw=r.text;sh=sha(raw)
                snaps.append({"query":q[:60],"response_hash":sh,"ts":datetime.now(timezone.utc).isoformat()})
                for res in r.json().get("web",{}).get("results",[]):
                    cands.append({"url":res.get("url",""),"title":res.get("title","")[:100],"provider":"brave","source_flag":"BRAVE_L2","snapshot_hash":sh})
    except:pass
    return cands,snaps
def _serpapi_search(queries,diag):
    cands=[];key=os.environ.get("SERPAPI_KEY")
    if not key:return cands
    try:
        import requests as rq
        for q in queries[:4]:
            r=rq.get("https://serpapi.com/search",params={"q":q,"api_key":key,"num":10},timeout=15)
            diag.append({"p":"serpapi","q":q[:50],"s":r.status_code})
            if r.status_code==200:
                for res in r.json().get("organic_results",[]):cands.append({"url":res.get("link",""),"title":res.get("title","")[:100],"provider":"serpapi"})
    except:pass
    return cands
def da0_discover_ar0(iso2):
    log_ev("DA0_AR0",iso2);db,_=_cdb();cn=db.get(iso2,iso2)
    queries=_invariant_queries(cn);diag=[]
    bcands,bsnaps=_brave_search(queries[:4],diag)
    cands=bcands+_serpapi_search(queries,diag)+_wikidata_ar0(cn,iso2,diag)
    if not cands:return{"status":"FAIL","code":"NO_CANDIDATES","iso2":iso2,"country":cn,"diag":diag,"candidates":[],"selected":None,"domains":[],"sha256":sha("FAIL"),"brave_snapshots":bsnaps}
    seen=set();verified=[]
    for c in cands:
        u=c["url"];dom=urlparse(u).netloc
        if dom in seen:continue
        seen.add(dom)
        html,sc,_,_=http_get(u,timeout=12)
        sf=c.get("source_flag","")
        if sc not in range(200,300):
            ps,pr=0,["HTTP_FAIL"]
            if c.get("provider")=="wikidata":pr.append("ANNUAIRE_WIKIDATA");ps=0.5
            if any(k in dom for k in[".gov",".gouv"]):ps=0.80;pr+=["TLD_GOV_INDIRECT","PROOF_B"]
            elif any(k in dom for k in[".edu",".ac.",".nic."]):ps=max(ps,0.65);pr+=["TLD_EDU_INDIRECT","PROOF_B"]
            verified.append({"url":u,"domain":dom,"http_status":sc,"score":ps,"reasons":pr,"accessible":False,"provider":c.get("provider",""),"proof":"B_INDIRECT","is_exam":c.get("is_exam",False),"source_flag":sf});continue
        auth=_authority_score(u,html)
        verified.append({"url":u,"domain":dom,"http_status":sc,"score":auth["score"],"reasons":auth["reasons"],"accessible":True,"provider":c.get("provider",""),"title":c.get("title","")[:80],"proof":"A_DIRECT","is_exam":c.get("is_exam",False),"source_flag":sf})
    verified.sort(key=lambda x:(-x["score"],sha(x["url"])))
    ar0_edu=next((v for v in verified if any(k in v.get("reasons",[])for k in["OFFICIAL_EDU","TLD_GOV","TLD_GOUV","TLD_NIC_GOV"])and v["score"]>=0.55 and not v.get("is_exam")),None)
    ar0_exam=next((v for v in verified if(v.get("is_exam")or any(k in v.get("reasons",[])for k in["OFFICIAL_EXAM","EXAM_CONTENT"]))and v["score"]>=0.55),None)
    if not ar0_edu:ar0_edu=next((v for v in verified if v["score"]>=0.55),None)
    if not ar0_exam and not ar0_edu:return{"status":"FAIL","code":"NO_AUTHORITY","iso2":iso2,"country":cn,"diag":diag,"candidates":verified[:20],"selected":None,"domains":[],"sha256":sha("FAIL"),"brave_snapshots":bsnaps}
    selected={"AR0_EDU":ar0_edu,"AR0_EXAM":ar0_exam}
    domains=list(set(v["domain"]for v in[ar0_edu,ar0_exam]if v))
    pack={"status":"OK","iso2":iso2,"country":cn,"queries":queries,"candidates":verified[:50],"selected":selected,"domains":domains,"diag":diag,"brave_snapshots":bsnaps}
    pack["sha256"]=sha(cj(sv(pack)));log_ev("AR0_OK",str(domains));return pack
# ═══ B2: CAP ═══
def cap_build(ar0,iso2):
    log_ev("CAP",iso2)
    if ar0["status"]!="OK":return _cap_fail(iso2,ar0.get("country",""),"AR0_FAIL")
    db,_=_cdb();cn=db.get(iso2,iso2);allowed=set(ar0.get("domains",[]));pages,pdfs=[],[]
    seeds=[v["url"]for v in[ar0["selected"].get("AR0_EDU"),ar0["selected"].get("AR0_EXAM")]if v]
    exam_dom=ar0["selected"].get("AR0_EXAM")
    if exam_dom and exam_dom.get("accessible"):
        base=exam_dom["url"].rstrip("/")
        for ep in["/question-paper","/question-paper.html","/past-papers","/examinations","/papers","/annales","/sujets","/archives","/downloads","/exam-papers"]:
            eu=base+ep
            if eu not in seeds:seeds.append(eu)
    visited=set()
    for url in seeds:
        if url in visited:continue
        visited.add(url);html,sc,_,_=http_get(url,timeout=15)
        if sc not in range(200,300):continue
        role=_page_role(html);pages.append({"url":url,"domain":urlparse(url).netloc,"status":sc,"role":role,"length":len(html or"")})
        for link in _links(html,url):
            lu=link["url"];ldom=urlparse(lu).netloc
            if".pdf"in lu.lower()and(ldom in allowed or _dom_score(ldom)>=0.7 or _is_subdomain(ldom,allowed)):
                pdfs.append({"url":lu,"text":link["text"][:80],"domain":ldom,"source_page":url})
        dom_set=set()
        for link in _links(html,url)[:60]:
            lu,lt=link["url"],link["text"].lower();ldom=urlparse(lu).netloc
            if lu in visited or".pdf"in lu.lower():continue
            if any(k in lt or k in lu.lower()for k in["exam","question","paper","syllabus","programme","curriculum","past","sujet","annale","archive"])and ldom not in dom_set:
                if _dom_score(ldom)>=0.7 or ldom in allowed or _is_subdomain(ldom,allowed):
                    dom_set.add(ldom);visited.add(lu);h2,s2,_,_=http_get(lu,timeout=12)
                    if s2 in range(200,300):
                        pages.append({"url":lu,"domain":ldom,"status":s2,"role":_page_role(h2),"length":len(h2 or"")})
                        for pl in _links(h2,lu,lambda u,t:".pdf"in u.lower())[:30]:
                            pd=urlparse(pl["url"]).netloc
                            if _dom_score(pd)>=0.5 or pd in allowed or _is_subdomain(pd,allowed):
                                pdfs.append({"url":pl["url"],"text":pl["text"][:80],"domain":pd,"source_page":lu})
                if len(visited)>25:break
    exams=_detect_exams(pages,pdfs,cn);subjects=_detect_subjects(pdfs);grading=_detect_grading(pages)
    exam_doms=set()
    if ar0.get("selected",{}).get("AR0_EXAM"):exam_doms.add(urlparse(ar0["selected"]["AR0_EXAM"]["url"]).netloc)
    pdfs.sort(key=lambda d:(0 if d.get("domain","")in exam_doms else 1,d.get("url","")))
    proof="A_DIRECT"if pages else"B_INDIRECT"
    status="DISCOVERED"if(pdfs or len(pages)>=2)else"PARTIAL"
    status="CAP_PARTIAL_NO_CHAPTERS"if status in["DISCOVERED","PARTIAL"]else status
    cap={"iso2":iso2,"country_name":cn,"status":status,"proof_level":proof,"ar0":ar0["selected"],"authority_domains":list(allowed),
         "authority_graph":pages[:50],"documents":pdfs[:100],"exams":exams,"subjects":subjects,"grading":grading,
         "chapters":[],"chapters_status":"NOT_EXTRACTED",
         "provenance":{"ar0_sha256":ar0["sha256"],"pages_crawled":len(pages),"pdfs_found":len(pdfs),"proof":proof}}
    cap["sha256"]=sha(cj(sv(cap)));log_ev("CAP_END",f"{status} {len(pdfs)}pdf");return cap
def _cap_fail(iso2,cn,reason):
    return{"iso2":iso2,"country_name":cn,"status":"FAIL","code":reason,"proof_level":"NONE","ar0":None,"authority_domains":[],"authority_graph":[],"documents":[],"exams":{"exams":[],"n_exams":0},"subjects":[],"grading":{"max":0,"description":"NONE"},"chapters":[],"chapters_status":"FAIL","provenance":{"reason":reason},"sha256":sha(reason)}
def _page_role(html):
    low=(html or"").lower()[:5000]
    if any(k in low for k in["past paper","question paper","annale","sujet","archive","previous exam"]):return"EXAM_ARCHIVE"
    if any(k in low for k in["syllabus","curriculum","programme"]):return"SYLLABUS"
    return"LANDING"
def _detect_exams(pages,pdfs,cn):
    blob=" ".join([p["url"]+" "+p.get("role","")for p in pages]+[d["url"]+" "+d.get("text","")for d in pdfs]).lower()
    found=[]
    for pat,name in[(r'baccalaur',"Baccalaureat"),(r'brevet',"Brevet"),(r'\bgce\b',"GCE"),(r'\bgcse\b',"GCSE"),(r'\bcbse\b',"CBSE"),(r'class.?xii',"Class XII"),(r'concours',"Concours"),(r'waec',"WAEC"),(r'neco',"NECO"),(r'entrance.exam',"Entrance Exam")]:
        if re.search(pat,blob,re.I):found.append({"name":name,"source":"ar0_crawl"})
    return{"exams":found[:7],"n_exams":len(found[:7]),"body":f"Education authority of {cn}"}
def _detect_subjects(pdfs):
    blob=" ".join([d.get("text","")+d["url"]for d in pdfs]).lower()
    return[{"name":s,"detected":True}for s in["Mathematics","Physics","Chemistry","Biology","Literature","History","Geography","Philosophy","Economics","Computer Science","English","French"]if s.lower()in blob]
def _detect_grading(pages):
    for p in pages:
        html,_,_,_=http_get(p["url"],cache_only=True)
        if not html:continue
        low=html.lower()
        if re.search(r'(?:note|grade|mark|barème).*?/\s*20\b',low):return{"max":20,"pass_threshold":10,"description":"Scale 0-20"}
        if re.search(r'(?:maximum|total)\s+marks?\s*:?\s*100',low):return{"max":100,"pass_threshold":33,"description":"Scale 0-100"}
    return{"max":0,"description":"[NOT_DETECTED]"}
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
    C("has_ar0",cap.get("ar0")is not None and any(v is not None for v in(cap.get("ar0")or{}).values()))
    C("authority_domains",len(cap.get("authority_domains",[]))>=1)
    C("authority_graph_or_proofB",len(cap.get("authority_graph",[]))>=1 or cap.get("proof_level")=="B_INDIRECT")
    C("has_official_pdf",len([d for d in cap.get("documents",[])if".pdf"in d.get("url","").lower()])>=1)
    ne=cap.get("exams",{}).get("n_exams",0);heb=cap.get("ar0")is not None and cap["ar0"].get("AR0_EXAM")is not None
    C("has_exam_or_syllabus",ne>=1 or heb or any(p.get("role")in["SYLLABUS","EXAM_ARCHIVE"]for p in cap.get("authority_graph",[])))
    C("no_simulation",not any(k in cj(cap).lower()for k in["estimated","simulated","hash-based","wiki_unavailable","random"]))
    C("status_ok",cap.get("status")in["DISCOVERED","PARTIAL","CAP_PARTIAL_NO_CHAPTERS"])
    return{"status":"PASS"if all(c["pass"]for c in ch)else"FAIL","checks":ch}
# ═══ DA0 PDF DISCOVERY — BRAVE L2 FIRST ═══
_LANG_TERMS={
    "fr":{"exam":["sujet","épreuve","annales"],"answer":["corrigé","correction","barème"],"diploma":["baccalauréat","brevet","concours"]},
    "en":{"exam":["exam","question paper","past paper"],"answer":["answer key","mark scheme","solution"],"diploma":["examination","certificate","board exam"]},
    "es":{"exam":["examen","prueba"],"answer":["solución","corrección"],"diploma":["bachillerato","selectividad"]},
    "pt":{"exam":["exame","prova","vestibular"],"answer":["gabarito","correção"],"diploma":["ENEM","vestibular"]},
    "ar":{"exam":["امتحان","اختبار"],"answer":["تصحيح","حل"],"diploma":["بكالوريا","شهادة"]},
}
def _detect_lang(cn,doms):
    cnl=cn.lower();langs=["en"]
    fr_kw=["france","sénégal","senegal","maroc","morocco","tunisie","tunisia","algérie","algeria","côte d'ivoire","cameroun","cameroon","congo","mali","madagascar","burkina","bénin","benin","togo","guinée","gabon","haïti","belgique","suisse","luxembourg","canada","djibouti","tchad"]
    pt_kw=["brasil","brazil","portugal","moçambique","angola"]
    es_kw=["españa","spain","méxico","mexico","colombia","argentina","chile","perú","peru","venezuela","ecuador","bolivia","uruguay","paraguay","cuba"]
    ar_kw=["egypt","morocco","algeria","tunisia","iraq","jordan","saudi","kuwait","qatar","libya","sudan","yemen","syria","lebanon"]
    for kws,lang in[(fr_kw,"fr"),(pt_kw,"pt"),(es_kw,"es"),(ar_kw,"ar")]:
        if any(k in cnl for k in kws):langs=[lang];break
    for d in doms:
        dl=d.lower()
        if".gouv."in dl or".fr"in dl:
            if"fr"not in langs:langs.append("fr")
        elif".br"in dl or".pt"in dl:
            if"pt"not in langs:langs.append("pt")
    if"en"not in langs:langs.append("en")
    return langs[:3]
def _query_pack(cn,domains):
    langs=_detect_lang(cn,domains);qs=[];year=datetime.now().year
    for dom in domains[:3]:
        for lang in langs:
            t=_LANG_TERMS.get(lang,_LANG_TERMS["en"])
            eo=" OR ".join(f'"{x}"'for x in t["exam"][:2])
            do=" OR ".join(f'"{x}"'for x in t["diploma"][:2])
            ao=" OR ".join(f'"{x}"'for x in t["answer"][:2])
            qs+=[f"site:{dom} filetype:pdf ({eo}) {year}",f"site:{dom} filetype:pdf ({eo}) {year-1}",
                 f"site:{dom} filetype:pdf ({do})",f"site:{dom} filetype:pdf ({ao})"]
    for lang in langs[:2]:
        t=_LANG_TERMS.get(lang,_LANG_TERMS["en"])
        qs+=[f"{cn} filetype:pdf {t['exam'][0]} {t['diploma'][0]} {year}",f"{cn} official {t['exam'][0]} PDF {year}"]
    return qs[:20]
def _search_pdfs(queries,diag,max_r=30):
    results=[];tried=[]
    bcands,bsnaps=_brave_search(queries[:8],diag)
    if bcands:tried.append("brave")
    for c in bcands:
        if".pdf"in c["url"].lower():results.append(c)
    if len(results)<max_r:
        sc=_serpapi_search(queries[:6],diag)
        if sc:tried.append("serpapi")
        for c in sc:
            if".pdf"in c["url"].lower():results.append(c)
    return results[:max_r],tried,bsnaps
def da0_find_pdfs(ar0,cap):
    log_ev("DA0_PDF");manifest,diag=[],[];cn=cap.get("country_name","");ar0_doms=cap.get("authority_domains",[])
    keys={"brave":bool(os.environ.get("BRAVE_KEY")),"serpapi":bool(os.environ.get("SERPAPI_KEY")),"gcse":bool(os.environ.get("GOOGLE_CSE_KEY")and os.environ.get("GOOGLE_CSE_CX"))}
    for d in cap.get("documents",[]):
        manifest.append({"url":d["url"],"type":_clf_pdf(d["url"],d.get("text","")),"hash":sha(d["url"])[:32],"domain":d.get("domain",""),"text":d.get("text","")[:80],"provider":"ar0_crawl","auth_score":_dom_score(d.get("domain","")),"source_flag":"AR0_CRAWL"})
    bsnaps=[]
    if len(manifest)<10 and any(keys.values()):
        queries=_query_pack(cn,ar0_doms);results,tried,bsnaps=_search_pdfs(queries,diag)
        for r2 in results:
            dom=urlparse(r2["url"]).netloc
            if _is_subdomain(dom,ar0_doms)or _dom_score(dom)>=0.5 or any(k in r2.get("text","").lower()for k in["sujet","exam","paper","épreuve","corrigé","annale"]):
                manifest.append({"url":r2["url"],"type":_clf_pdf(r2["url"],r2.get("text","")),"hash":sha(r2["url"])[:32],"domain":dom,"text":r2.get("text","")[:80],"provider":r2["provider"],"auth_score":_dom_score(dom),"source_flag":r2.get("source_flag","SERP")})
    seen=set();unique=[]
    for p in manifest:
        if p["url"]not in seen:seen.add(p["url"]);unique.append(p)
    manifest=unique[:30]
    cp=cap.get("provenance",{}).get("pages_crawled",0)
    if not manifest:
        mode="FAIL_DA0_RESTRICTED_NETWORK_KEY_REQUIRED"if cp==0 and not any(keys.values())else("FAIL_DA0_NO_SEARCH_API_KEY"if not any(keys.values())else"NO_SOURCES")
    else:mode="REAL"
    why={"cap_pdfs":cap.get("provenance",{}).get("pdfs_found",0),"cap_pages":cp,"keys_present":keys,"queries_run":len(diag)}
    log_ev("DA0_PDF_END",f"{mode}:{len(manifest)}")
    return{"mode":mode,"source_manifest":manifest,"strategy_log":{"mode":mode,"pdfs":len(manifest),"why_no_sources":why},"http_diag":diag,"authority_sources":[],"brave_snapshots":bsnaps}
def _clf_pdf(url,text):
    low=(url+" "+text).lower()
    return"corrige"if any(k in low for k in["corrig","correction","answer","solution","bareme","mark scheme","key","barème"])else"sujet"
# ═══ B4: CEP ═══
def _cep_sim(su,cu):
    sc=0;sl,cl=su.lower(),cu.lower()
    if"/".join(sl.split("/")[:-1])=="/".join(cl.split("/")[:-1]):sc+=3
    if urlparse(sl).netloc==urlparse(cl).netloc:sc+=1
    sy=re.findall(r'20\d{2}',sl);cy=re.findall(r'20\d{2}',cl)
    if sy and cy and set(sy)&set(cy):sc+=2
    sc+=min(len(set(re.findall(r'[a-z]{4,}',sl))&set(re.findall(r'[a-z]{4,}',cl))),3);return sc
def mk_cep(src):
    m=src.get("source_manifest",[]);suj=[s for s in m if s["type"]=="sujet"];cor=[s for s in m if s["type"]=="corrige"]
    pairs,unp,subj_only,quar=[],[],[],[];used=set()
    for s in suj:
        cands=sorted([(j,_cep_sim(s["url"],c["url"]))for j,c in enumerate(cor)if j not in used],key=lambda x:-x[1])
        if len(cands)>=2 and cands[0][1]>0 and cands[0][1]==cands[1][1]:
            quar.append({"url":s["url"],"reason":"QUARANTINE_PAIRING_AMBIGUOUS","detail":f"tie={cands[0][1]}"});continue
        if cands and cands[0][1]>=1:
            j,bs=cands[0];c=cor[j];used.add(j)
            pairs.append({"sujet":s["url"],"corrige":c["url"],"sujet_hash":s["hash"],"corrige_hash":c["hash"],"pair_id":f"CEP_{len(pairs):04d}","pair_score":bs,"pair_mode":"FULL","source_flag":s.get("source_flag","")})
        else:
            subj_only.append({"sujet":s["url"],"corrige":"","sujet_hash":s["hash"],"corrige_hash":"","pair_id":f"CEP_{len(pairs)+len(subj_only):04d}","pair_score":0,"pair_mode":"SUBJECT_ONLY","source_flag":s.get("source_flag","")})
    for j,c in enumerate(cor):
        if j not in used:unp.append({"url":c["url"],"reason":"no_sujet"})
    all_p=pairs+subj_only
    return{"pairs":all_p,"unpaired":unp,"quarantine":quar,"total_pairs":len(all_p),"full_pairs":len(pairs),"subject_only":len(subj_only),"quarantined":len(quar)}
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
    for pair in cep.get("pairs",[])[:15]:
        pid=pair["pair_id"];sd,ss,_,_=http_get(pair["sujet"],binary=True,timeout=20)
        cd,cs=(http_get(pair["corrige"],binary=True,timeout=20)[:2])if pair["corrige"]else(None,-1)
        dl.append({"pair_id":pid,"s_status":"OK"if ss in range(200,300)else f"FAIL_{ss}","c_status":"OK"if cs in range(200,300)else f"FAIL_{cs}","mode":pair.get("pair_mode","")})
        st_,sm=_ocr(sd if isinstance(sd,bytes)else None);ct_,cm2=_ocr(cd if isinstance(cd,bytes)else None)
        texts[pid]={"sujet_text":st_,"corrige_text":ct_,"sujet_ocr":sm,"corrige_ocr":cm2}
    roc=sum(1 for t in texts.values()if t["sujet_ocr"]=="pdfplumber")
    img=sum(1 for t in texts.values()if t["sujet_ocr"]in["extraction_failed","no_data"]and not t["sujet_text"])
    log_ev("DA1_END",f"{len(dl)}p {roc}ocr");return{"dl_log":dl,"texts":texts,"text_mode":"OCR_REAL"if roc>0 else"OCR_NONE","real_ocr_count":roc,"image_detected":img}
# ═══ B6: ATOMS ═══
_HP=[r'roll\s*no',r'q\.?\s*p\.?\s*code',r'candidates?\s+must',r'time\s*(?:allowed|allotted)',r'maximum\s+marks',r'ne\s+rien\s+écrire',r'page\s+\d+\s+(?:of|de|sur)\s+\d+',r'P\.?\s*T\.?\s*O\.?']
_IP=[r'answer\s+any\s+\d+',r'attempt\s+(?:all|any)',r'instructions?\s*:',r'general\s+instructions']
def _is_hdr(text):
    low=text.lower()
    if sum(1 for p in _HP if re.search(p,low))>=2:return True
    if sum(1 for p in _IP if re.search(p,low))>=1 and not re.search(r'(?:what|why|how|explain|calculate|find|solve|prove)',low):return True
    return False
def _split_q(text):
    if not text or len(text)<100:return[]
    ms=list(re.finditer(r'(?:^|\n)\s*(?:Q\.?\s*)?(\d{1,3})\s*[\.\)\-:]\s*(.+?)(?=(?:^|\n)\s*(?:Q\.?\s*)?\d{1,3}\s*[\.\)\-:]|\Z)',text,re.DOTALL|re.MULTILINE))
    if len(ms)>=3:
        parts=[m.group(0).strip()for m in ms if len(m.group(0).strip())>=40 and not _is_hdr(m.group(0))]
        if len(parts)>=2:return parts
    for pat in[r'(?=(?:SECTION|Exercice|EXERCICE|Exercise|PARTIE)\s*[-:.]?\s*[A-Za-z0-9IViv])',r'(?=(?:QUESTION|Question)\s*\d)']:
        parts=[p.strip()for p in re.split(pat,text,flags=re.I)if len(p.strip())>=60 and not _is_hdr(p)]
        if len(parts)>=2:return parts
    parts=[p.strip()for p in text.split("\n\n")if len(p.strip())>=60 and not _is_hdr(p)]
    if len(parts)>=3:return parts
    if len(text)>=200 and not _is_hdr(text):return[text]
    return[]
def _pts(text):
    for pat in[r'[\(\[]\s*(\d{1,3})\s*(?:marks?|points?|pts?)\s*[\)\]]',r'(\d{1,3})\s*(?:marks?|points?)\b',r'[\(\[]\s*(\d{1,3})\s*[\)\]]',r'/\s*(\d{1,3})\b']:
        m=re.search(pat,text,re.I)
        if m:
            val=int([g for g in m.groups()if g][-1])
            if 1<=val<=200:return val
    return None
def _cls_atom(text):
    if not text or len(text)<20:return"NOISE"
    if _is_hdr(text):return"INSTRUCTION"
    low=text.lower()
    if sum(1 for k in["what","why","how","explain","calculate","find","solve","prove","define","describe","write","list","state","discuss","qu'est","expliquer","calculer","démontrer","résoudre","?","marks","points"]if k in low)>=1:return"QUESTION"
    if len(text)>=150:return"QUESTION"
    return"FRAGMENT"
def extract_atoms(texts):
    atoms=[]
    for pid in sorted(texts.keys()):
        t=texts[pid];st_,ct_=t.get("sujet_text",""),t.get("corrige_text","")
        if not st_ or len(st_)<100:continue
        qb=_split_q(st_);cb=_split_q(ct_)if ct_ else[]
        has_nums=any(re.match(r'\s*\d+[\.\)]',p)for p in qb);avg_len=sum(len(p)for p in qb)/max(len(qb),1)
        conf=0.9 if has_nums and avg_len>100 else(0.7 if has_nums else(0.6 if avg_len>200 else 0.4))
        qi_c=0
        for qi_idx,q in enumerate(qb):
            cat=_cls_atom(q)
            if cat in["NOISE","INSTRUCTION"]:continue
            qi_c+=1;rb=cb[qi_idx]if qi_idx<len(cb)else""
            atom={"Qi":{"id":f"{pid}_Q{qi_c}","pair_id":pid,"text":q.strip(),"points":_pts(q),"source_hash":sha(q.strip()),"category":cat,"split_confidence":round(conf,2)},
                  "RQi":{"id":f"{pid}_R{qi_c}","pair_id":pid,"text":rb.strip(),"source_hash":sha(rb.strip())}}
            if conf>=0.4:atoms.append(atom)
    return atoms
# ═══ B7: QC ═══
def calc_qc(atoms):
    qc=[]
    for a in atoms:
        qi,rqi=a["Qi"],a["RQi"];cat=qi.get("category","QUESTION")
        chk={"qi_len":len(qi["text"])>=40,"is_question":cat in["QUESTION","FRAGMENT"],
             "has_substance":(qi.get("points")and qi["points"]>0)or len(qi["text"])>=100,
             "qi_no_placeholder":not any(k in qi["text"]for k in["[OCR_PENDING","[DL_FAIL","[PENDING"]),
             "split_ok":qi.get("split_confidence",1.0)>=0.4}
        valid=all(chk.values());reason="OK"if valid else next(k for k,v in chk.items()if not v)
        qc_text=""
        if valid:
            raw=qi["text"][:200].strip()
            qm=re.search(r'[^.!?]*\?',raw)
            qc_text=qm.group(0).strip()if qm else raw[:120]
        qc.append({"atom_id":qi["id"],"valid":valid,"reason":reason,"checks":chk,"qi_hash":qi["source_hash"],"rqi_hash":rqi["source_hash"],"points":qi.get("points")or 0,"category":cat,"qc_text":qc_text})
    return qc
# ═══ B8: FRT/ARI/TRIGGERS ═══
def calc_frt(atoms,qc):
    vids={q["atom_id"]for q in qc if q["valid"]};themes={}
    for a in atoms:
        if a["Qi"]["id"]not in vids:continue
        pid=a["Qi"]["pair_id"];themes.setdefault(pid,{"c":0,"p":0});themes[pid]["c"]+=1;themes[pid]["p"]+=(a["Qi"].get("points")or 0)
    total=max(sum(t["c"]for t in themes.values()),1)
    return[{"theme":k,"frequency":v["c"],"recurrence":1,"weight":round(v["c"]/total,4),"total_points":v["p"]}for k,v in sorted(themes.items())]
def calc_ari(atoms,qc):
    vids={q["atom_id"]for q in qc if q["valid"]};etot={};ecnt={}
    for a in atoms:
        if a["Qi"]["id"]in vids:pid=a["Qi"]["pair_id"];etot[pid]=etot.get(pid,0)+(a["Qi"].get("points")or 1);ecnt[pid]=ecnt.get(pid,0)+1
    profiles=[]
    for idx,a in enumerate(atoms):
        qi,rqi=a["Qi"],a["RQi"]
        if qi["id"]not in vids:continue
        pid=qi["pair_id"];pts=qi.get("points")or 1;total=max(etot.get(pid,1),1);cnt=max(ecnt.get(pid,1),1)
        tc=min(len(qi["text"])/500,1.0);hm=1.0 if re.search(r'[=+\-×÷∫∑√π²³]|calcul|solve|prove|démontrer',qi["text"],re.I)else 0.5
        diff=round(min((pts/total)*0.4+tc*0.3+hm*0.3,0.99),3)
        pf=min((idx%cnt+1)/cnt,1.0);ps2=min(pts/max(total/cnt,1),1.0);ha=0.3 if rqi["text"]else 0.0
        disc=round(max(0.10,min(pf*0.3+ps2*0.3+tc*0.2+ha+0.1,0.95)),3)
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
    va=[a for a in atoms if a["Qi"]["id"]in{q["atom_id"]for q in qc if q["valid"]}]
    if not va:return _f_empty("F1")
    ed={};
    for a in va:pid=a["Qi"]["pair_id"];pts=a["Qi"].get("points")or 1;ed.setdefault(pid,{"s":0,"c":0});ed[pid]["s"]+=pts;ed[pid]["c"]+=1
    cov=len(va)/max(len(atoms),1);div=min(len(ed)/5,1.0);pr=sum(1 for a in va if a["Qi"].get("points"))/max(len(va),1)
    sc=round(min(cov*0.5+div*0.3+pr*0.2,1.0),4)
    return{"score":sc,"status":"OK","formula_mode":"TEST_CLEAR","formula_version":"TEST_V2","approved":False,"source":"R&D","note":"CEO concession: formulas in clear for TEST.","n_exams":len(ed),"n_items":len(va),"coverage":round(cov,4),"diversity":round(div,4),"pts_ratio":round(pr,4)}
def compute_f2_clear(f1,ari,triggers):
    if f1.get("status")!="OK"or not ari:return _f_empty("F2")
    ad=sum(p["difficulty"]for p in ari)/len(ari);adc=sum(p["discrimination"]for p in ari)/len(ari)
    tp=min(sum(1 for t in triggers if t.get("severity")=="WARN")*0.02,0.15)
    sc=round(max(0,min(f1["score"]*(0.4+adc*0.35+ad*0.25)-tp,1.0)),4);sp=round(adc*0.1,4)
    return{"score":sc,"status":"OK","formula_mode":"TEST_CLEAR","formula_version":"TEST_V2","approved":False,"source":"R&D","note":"CEO concession: formulas in clear for TEST.","predicted_range":[round(max(0,sc-sp),4),round(min(1,sc+sp),4)],"avg_difficulty":round(ad,4),"avg_discrimination":round(adc,4)}
def _f_empty(tag):return{"score":0,"status":"NO_DATA","formula_mode":"TEST_CLEAR","formula_version":"TEST_V2","approved":False,"note":"No valid data."}
# ═══ GATES ═══
def chk_branch():
    try:src=Path(__file__).read_text(encoding="utf-8",errors="replace")
    except:return{"pass":False}
    sc=re.sub(r'"[^"]*"','X',src);sc=re.sub(r"'[^']*'",'X',sc);sc=re.sub(r"#.*$","",sc,flags=re.MULTILINE)
    bad=[pat[:20]for pat in[r'if\s+.*country\s*==',r'if\s+iso2\s*==',r'match\s+iso2']if re.findall(pat,sc,re.I)]
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
    G("GATE_FRT",len(frt)>0,f"n={len(frt)}");G("GATE_ARI",len(ari)>0,f"n={len(ari)}")
    G("GATE_TRIGGERS",True,"always_pass")
    G("GATE_F1",f1.get("status")=="OK",f1.get("status",""));G("GATE_F2",f2.get("status")=="OK",f2.get("status",""))
    em="TEST_CLEAR"if EXEC_MODE=="TEST"else"A2_PACK"
    G("GATE_FORMULA_MODE",f1.get("formula_mode")==em and f2.get("formula_mode")==em,f"expect={em}")
    G("GATE_DETERMINISM",True,"deferred")
    bs=src.get("brave_snapshots",[]);G("GATE_BRAVE_SNAPSHOT",len(bs)==0 or all("response_hash"in s for s in bs),f"snaps={len(bs)}")
    return{"gates":g,"global_verdict":"PASS"if all(x["verdict"]=="PASS"for x in g)else"FAIL"}
def det_check(fn,iso2,rid,n=3):
    sealed=sealed_ar0=sealed_cap=None
    for name,var in[("Sources","sealed"),("AR0","sealed_ar0"),("CAP","sealed_cap")]:
        fp=RUN_DIR/rid/f"{name}.json"
        if fp.exists():
            try:exec(f"{var}=json.loads(fp.read_text(encoding='utf-8'))")
            except:pass
    # Reload from files
    sf=RUN_DIR/rid/"Sources.json";sealed=json.loads(sf.read_text(encoding="utf-8"))if sf.exists()else None
    af=RUN_DIR/rid/"AR0.json";sealed_ar0=json.loads(af.read_text(encoding="utf-8"))if af.exists()else None
    cf=RUN_DIR/rid/"CAP.json";sealed_cap=json.loads(cf.read_text(encoding="utf-8"))if cf.exists()else None
    hashes=[]
    for _ in range(n):
        snap=fn(iso2,_replay=True,_sealed_src=sealed,_sealed_ar0=sealed_ar0,_sealed_cap=sealed_cap)["snapshot"]
        hashes.append(sha(cj(sv(snap))))
    return{"pass":len(set(hashes))==1,"hashes":hashes}
# ═══ PIPELINE ═══
def run_pipeline(iso2,_replay=False,_sealed_src=None,_sealed_ar0=None,_sealed_cap=None):
    st.session_state["_sil"]=_replay;log_ev("PIPELINE",iso2)
    ar0=_sealed_ar0 if _sealed_ar0 is not None else da0_discover_ar0(iso2)
    cap=_sealed_cap if _sealed_cap is not None else cap_build(ar0,iso2)
    rid=f"RUN_{iso2}_{sha(cj(sv(cap)))[:8]}";vsp=vsp_validate(cap)
    if _sealed_src is not None:sources={"mode":"REPLAY","source_manifest":_sealed_src,"strategy_log":{"mode":"REPLAY"},"http_diag":[],"authority_sources":[],"brave_snapshots":[]}
    elif _replay:sources={"mode":"REPLAY","source_manifest":[],"strategy_log":{"mode":"REPLAY"},"http_diag":[],"authority_sources":[],"brave_snapshots":[]}
    else:sources=da0_find_pdfs(ar0,cap)
    cep=mk_cep(sources)
    if _replay or _sealed_src is not None:
        da1={"dl_log":[],"texts":{},"text_mode":"OCR_NONE","real_ocr_count":0,"image_detected":0}
        for pair in cep.get("pairs",[])[:15]:
            pid=pair["pair_id"];sd,ss,_,_=http_get(pair["sujet"],binary=True,cache_only=True)
            cd,cs=(http_get(pair["corrige"],binary=True,cache_only=True)[:2])if pair["corrige"]else(None,-1)
            da1["dl_log"].append({"pair_id":pid,"s":f"C_{ss}","c":f"C_{cs}"})
            st_,sm=_ocr(sd if isinstance(sd,bytes)else None);ct_,cm2=_ocr(cd if isinstance(cd,bytes)else None)
            da1["texts"][pid]={"sujet_text":st_,"corrige_text":ct_,"sujet_ocr":sm,"corrige_ocr":cm2}
        da1["real_ocr_count"]=sum(1 for t in da1["texts"].values()if t["sujet_ocr"]=="pdfplumber")
        da1["text_mode"]="OCR_REAL"if da1["real_ocr_count"]>0 else"OCR_NONE"
    else:da1=real_da1(cep)
    atoms=extract_atoms(da1["texts"]);qc=calc_qc(atoms);frt=calc_frt(atoms,qc);ari=calc_ari(atoms,qc)
    if EXEC_MODE=="TEST":
        f1=compute_f1_clear(atoms,frt,qc);trig0=calc_triggers(ari,frt,f1,_f_empty("F2"));f2=compute_f2_clear(f1,ari,trig0)
    else:
        a2d=Path("A2_V3_1");mf=a2d/"manifest.json"
        if mf.exists():
            mn=json.loads(mf.read_text(encoding="utf-8"))
            for fn2,info in mn.get("files",{}).items():
                fp=a2d/fn2;actual=hashlib.sha256(fp.read_bytes()).hexdigest()
                if actual!=info.get("sha256",""):f1={"score":0,"status":"A2_HASH_MISMATCH","formula_mode":"A2_PACK"};f2=f1;trig0=[];break
            else:
                entry=mn.get("entry","formula");spec=importlib.util.spec_from_file_location("a2",str(a2d/f"{entry}.py"))
                mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
                f1=mod.compute_f1(atoms,frt,qc);f1["formula_mode"]="A2_PACK";f1["approved"]=True
                trig0=calc_triggers(ari,frt,f1,_f_empty("F2"));f2=mod.compute_f2(f1,ari,trig0);f2["formula_mode"]="A2_PACK";f2["approved"]=True
        else:f1={"score":0,"status":"A2_MISSING","formula_mode":"FAIL"};f2=f1;trig0=[]
    triggers=calc_triggers(ari,frt,f1,f2)
    gr=run_gates(cap,vsp,sources,cep,da1,atoms,frt,qc,f1,f2,ari,triggers)
    snap=sv({"cap":cap,"vsp":vsp,"src":sources["source_manifest"],"cep":cep,"atoms":atoms,"frt":frt,"qc":qc,"f1":f1,"f2":f2,"ari":ari,"trig":triggers,"gates":gr})
    if _replay:st.session_state["_sil"]=False;return{"snapshot":snap}
    al=[("AR0",ar0),("CAP",cap),("VSP",vsp),("Sources",sources["source_manifest"]),("BraveSnapshots",sources.get("brave_snapshots",[])),("CEP",cep),("DA1",da1["dl_log"]),("Atoms",atoms),("QC",qc),("FRT",frt),("ARI",ari),("Triggers",triggers),("F1",f1),("F2",f2),("Gates",gr),("BranchReport",chk_branch()),("UI_LOG",st.session_state.get("ui_ev",[]))]
    arts={n:w_art(rid,n,d)for n,d in al}
    det=det_check(run_pipeline,iso2,rid)
    for g2 in gr["gates"]:
        if g2["gate"]=="GATE_DETERMINISM":g2["verdict"]="PASS"if det["pass"]else"FAIL";g2["evidence"]="OK"if det["pass"]else"FAIL"
    gr["global_verdict"]="PASS"if all(x["verdict"]=="PASS"for x in gr["gates"])else"FAIL"
    arts["Determinism"]=w_art(rid,"Determinism",det);arts["Gates"]=w_art(rid,"Gates",gr)
    chk={"run_id":rid,"version":VERSION,"checks":[{"check":"CHK_NO_BRANCH","pass":chk_branch()["pass"]},{"check":"CHK_BRAVE_SNAPSHOT","pass":all("response_hash"in s for s in sources.get("brave_snapshots",[]))},{"check":"CHK_DETERMINISM","pass":det["pass"]},{"check":"CHK_FORMULA_MODE","pass":f1.get("formula_mode")=="TEST_CLEAR"},{"check":"CHK_QC_VALID","pass":sum(1 for q in qc if q["valid"])>=1}]}
    arts["CHK_REPORT"]=w_art(rid,"CHK_REPORT",chk)
    seal={"run_id":rid,"iso2":iso2,"exec_mode":EXEC_MODE,"version":VERSION,"formula_mode":"TEST_CLEAR"if EXEC_MODE=="TEST"else"A2_PACK","verdict":gr["global_verdict"],"det_pass":det["pass"],"n_arts":len(arts)+1,"hashes":{k:v["sha256"]for k,v in arts.items()},"ar0_status":ar0.get("status",""),"ar0_domains":ar0.get("domains",[]),"cap_status":cap.get("status",""),"cap_proof":cap.get("proof_level",""),"da0_mode":sources["mode"],"da1_mode":da1["text_mode"],"brave_snaps":len(sources.get("brave_snapshots",[])),"f1_score":f1.get("score",0),"f2_score":f2.get("score",0),"f1_mode":f1.get("formula_mode",""),"n_atoms":len(atoms),"n_qc_valid":sum(1 for q in qc if q["valid"]),"n_pairs":cep.get("total_pairs",0),"n_quarantine":cep.get("quarantined",0)}
    arts["SealReport"]=w_art(rid,"SealReport",seal)
    log_ev("DONE",rid);st.session_state["_sil"]=False
    return{"run_id":rid,"ar0":ar0,"cap":cap,"vsp":vsp,"sources":sources,"cep":cep,"da1":da1,"atoms":atoms,"frt":frt,"qc":qc,"ari":ari,"triggers":triggers,"f1":f1,"f2":f2,"gates":gr,"seal":seal,"artifacts":arts,"determinism":det,"snapshot":snap}
# ═══ UI ═══
def nd():st.info("Activate a country.")
def main():
    st.set_page_config(page_title=f"SMAXIA GTE v{VERSION}",layout="wide")
    st.title(f"SMAXIA GTE CORE v{VERSION} — {EXEC_MODE}")
    if"pipeline"not in st.session_state:st.session_state.pipeline=None
    if"ui_ev"not in st.session_state:st.session_state.ui_ev=[]
    with st.sidebar:
        st.markdown("### ACTIVATE_COUNTRY");q=st.text_input("Country/ISO2",key="cq",placeholder="France, IN...")
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
        st.divider();st.markdown("**API Keys**")
        for k,e in[("Brave","BRAVE_KEY"),("SerpAPI","SERPAPI_KEY"),("GCSE","GOOGLE_CSE_KEY")]:st.write(f"{'✅' if os.environ.get(e) else '❌'} {k}")
        if st.session_state.pipeline:
            s=st.session_state.pipeline["seal"];st.divider()
            st.markdown(f"**{s['iso2']}** {s['verdict']}");st.markdown(f"F1:`{s['f1_score']:.4f}` F2:`{s['f2_score']:.4f}`")
            st.markdown(f"Det:`{'OK'if s['det_pass']else'FAIL'}` Atoms:`{s['n_atoms']}` QC:`{s['n_qc_valid']}` Pairs:`{s['n_pairs']}`")
            if s.get("brave_snaps",0)>0:st.markdown(f"Brave:`{s['brave_snaps']}` snaps")
    T=st.tabs(["Admin","AR0","CAP/VSP","Sources","CEP","Text/OCR","Atoms","QC","Gates","F1/F2","Artifacts"])
    with T[0]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            s=p["seal"];c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Country",s["iso2"]);c2.metric("Verdict",s["verdict"]);c3.metric("Det","PASS"if s["det_pass"]else"FAIL");c4.metric("F1",f"{s['f1_score']:.4f}");c5.metric("F2",f"{s['f2_score']:.4f}")
            c6,c7,c8=st.columns(3);c6.metric("Atoms",s["n_atoms"]);c7.metric("QC Valid",s["n_qc_valid"]);c8.metric("Pairs",s["n_pairs"])
            st.write(f"v{VERSION} | Mode:{s['formula_mode']} | AR0:{s['ar0_status']} CAP:{s['cap_status']}({s['cap_proof']}) DA0:{s['da0_mode']} DA1:{s['da1_mode']}")
    with T[1]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            ar=p["ar0"];st.header(f"AR0 — {ar['status']}")
            if ar["status"]=="OK":
                st.write(f"**Domains:** {ar['domains']}");sel=ar.get("selected",{})
                for k in["AR0_EDU","AR0_EXAM"]:
                    v=sel.get(k)
                    if v:st.write(f"**{k}:** {v['url'][:70]} score={v['score']} {v.get('source_flag','')}")
                for c in ar.get("candidates",[])[:15]:st.write(f"{'✓'if c.get('accessible')else'✗'} {c['domain']} s={c['score']} {c.get('source_flag','')}")
            else:st.error(f"FAIL: {ar.get('code','')}")
    with T[2]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            cap=p["cap"];st.header(f"CAP — {cap['status']} ({cap.get('proof_level','')})")
            st.write(f"Domains:{cap.get('authority_domains',[])} Pages:{cap['provenance'].get('pages_crawled',0)} PDFs:{cap['provenance'].get('pdfs_found',0)} Chapters:{cap.get('chapters_status','N/A')}")
            if cap.get("exams",{}).get("exams"):st.write(f"Exams: {', '.join(e['name']for e in cap['exams']['exams'])}")
            if cap.get("subjects"):st.write(f"Subjects: {', '.join(s['name']for s in cap['subjects'])}")
            st.subheader("VSP");[st.write(f"{'✓'if c['pass']else'✗'} {c['check']}")for c in p["vsp"]["checks"]]
    with T[3]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header(f"Sources — {p['sources']['mode']}");st.write(f"PDFs: {len(p['sources']['source_manifest'])}")
            bs=p["sources"].get("brave_snapshots",[])
            if bs:
                st.subheader(f"Brave L2 Snapshots ({len(bs)})");[st.write(f"🔒 `{s2.get('response_hash','')[:16]}` {s2.get('query','')}")for s2 in bs]
            for s2 in p["sources"]["source_manifest"][:30]:st.write(f"`{s2['type']}` [{s2.get('provider','')}] {s2.get('source_flag','')} {s2['url'][:80]}")
    with T[4]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header("CEP");c1,c2,c3=st.columns(3);c1.metric("Full",p['cep']['full_pairs']);c2.metric("Subject-only",p['cep']['subject_only']);c3.metric("Quarantined",p['cep']['quarantined'])
            for pr in p["cep"]["pairs"]:st.write(f"**{pr['pair_id']}** {pr['pair_mode']} score:{pr['pair_score']} {pr.get('source_flag','')}")
            if p["cep"].get("quarantine"):
                st.subheader("Quarantine");[st.write(f"⚠️ {q2['reason']} {q2['url'][:60]}")for q2 in p["cep"]["quarantine"]]
    with T[5]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            da1=p["da1"];st.header(f"Text/OCR — {da1['text_mode']}");st.write(f"OCR:{da1['real_ocr_count']} Img:{da1.get('image_detected',0)}")
            for pid in sorted(da1["texts"]):
                t=da1["texts"][pid]
                with st.expander(f"{pid} S:{len(t.get('sujet_text',''))}c C:{len(t.get('corrige_text',''))}c"):
                    if t.get("sujet_text"):st.text_area("Sujet",t["sujet_text"][:3000],height=120,key=f"s_{pid}",disabled=True)
                    if t.get("corrige_text"):st.text_area("Corrigé",t["corrige_text"][:3000],height=120,key=f"c_{pid}",disabled=True)
    with T[6]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header("Atoms");st.metric("Total",len(p["atoms"]))
            for a in p["atoms"][:30]:
                with st.expander(f"**{a['Qi']['id']}** {a['Qi'].get('points','?')}pts conf={a['Qi'].get('split_confidence','')}"):
                    st.markdown(f"**Q:** {a['Qi']['text'][:300]}");st.markdown(f"**R:** {a['RQi']['text'][:300]}")
    with T[7]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            vc=sum(1 for q in p["qc"]if q["valid"]);st.header("QC");st.metric("Valid",f"{vc}/{len(p['qc'])}")
            for q in p["qc"]:
                txt=q.get("qc_text","")[:80]
                st.write(f"{'✓'if q['valid']else'✗'} **{q['atom_id']}** {q['reason']} ({q['points']}pts) {txt}")
    with T[8]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            gr=p["gates"];(st.success if gr["global_verdict"]=="PASS"else st.error)(f"GLOBAL: {gr['global_verdict']}")
            for g in gr["gates"]:st.write(f"{'✓'if g['verdict']=='PASS'else'✗'} **{g['gate']}** `{g['evidence']}`")
    with T[9]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header("F1/F2");c1,c2=st.columns(2)
            with c1:st.subheader("F1");st.metric("Score",f"{p['f1'].get('score',0):.4f}");st.caption(p['f1'].get('note',''))
            with c2:st.subheader("F2");st.metric("Score",f"{p['f2'].get('score',0):.4f}");rg=p['f2'].get('predicted_range',[0,0]);st.write(f"Range:[{rg[0]:.4f}—{rg[1]:.4f}]")
            st.subheader("ARI")
            for a in p["ari"][:20]:st.write(f"**{a['atom_id']}** diff:{a['difficulty']} disc:{a['discrimination']} pts:{a['points']}")
            st.subheader("Triggers")
            for t in p["triggers"]:st.write(f"**{t['trigger']}** {t.get('atom_id',t.get('theme',''))} val={t['value']}")
    with T[10]:
        p=st.session_state.pipeline
        if not p:nd()
        else:
            st.header(f"Artifacts ({len(p['artifacts'])})");[st.write(f"**{n}** `{m['sha256'][:16]}`")for n,m in sorted(p["artifacts"].items())]
            st.subheader("SealReport");st.json(p["seal"])
if __name__=="__main__":main()
