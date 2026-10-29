const KIND_EMOJI={text:'📄',image:'🖼️',table:'📊',plan:'📐',email:'✉️'};
const KIND_LABEL={text:'Texte',image:'Image',table:'Tableau',plan:'Plan',email:'E-mail'};
async function api(path,opts){
  const r=await fetch(path,opts);
  if(!r.ok){let m=r.statusText;try{m=(await r.json()).detail||m}catch(e){}throw new Error(m)}
  return r.json();
}
function el(tag,attrs={},...kids){
  const e=document.createElement(tag);
  for(const[k,v]of Object.entries(attrs)){k==='class'?e.className=v:k.startsWith('on')?e.addEventListener(k.slice(2),v):e.setAttribute(k,v)}
  for(const c of kids.flat())e.append(c instanceof Node?c:document.createTextNode(c));
  return e;
}
function nav(on){
  document.querySelector('header').append(
    el('b',{},'RAG-Anything'),
    el('a',{href:'/',class:on==='search'?'on':''},'Recherche'),
    el('a',{href:'/ingest',class:on==='ingest'?'on':''},'Ingestion'));
}
