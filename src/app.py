import base64
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import anthropic as _anthropic_sdk
import bcrypt
import pymupdf as fitz
import sqlalchemy as sa
import streamlit as st
import streamlit.components.v1 as _components
from PIL import Image as _PILImage
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.supabase import SupabaseVectorStore
from supabase import Client, create_client


# ── Percorsi e .env ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logo incorporato come base64 — evita dipendenze da static serving
_LOGO_PATH = BASE_DIR / "static" / "logo.png"
_LOGO_SRC = (
    "data:image/png;base64,"
    + base64.b64encode(_LOGO_PATH.read_bytes()).decode()
)

BUCKET = "cga-documents"

CATEGORIES: dict[str, str] = {
    "Auto e Moto":    "auto",
    "Casa":           "casa",
    "Vita e Persona": "vita",
    "Viaggi":         "viaggi",
}

CATEGORY_ICONS: dict[str, str] = {
    "Auto e Moto":    "🚗",
    "Casa":           "🏠",
    "Vita e Persona": "❤️",
    "Viaggi":         "✈️",
}

TIPOLOGIE: list[str] = ["Set Informativo", "Circolare", "Normativa"]

TIPOLOGIA_BADGE_COLOR: dict[str, str] = {
    "Set Informativo": "#8B2061",
    "Circolare":       "#F47920",
    "Normativa":       "#1565C0",
}

SYSTEM_PROMPT = (
    "Sei l'Oracolo delle Polizze dell'Agenzia Sara Assicurazioni. "
    "Stai parlando con un agente assicurativo o un impiegato, NON con un cliente finale. "
    "Le tue risposte devono essere tecniche, precise e operative. "
    "Cita sempre il nome del documento e la pagina. "
    "Usa linguaggio professionale da operatore del settore. "
    "Se la risposta ha implicazioni per il cliente finale, segnalalo come nota separata. "
    "Se non trovi l'informazione, dì esplicitamente che non è presente nei documenti. "
    "Non inventare mai coperture o esclusioni."
)

SPLITTER = SentenceSplitter(chunk_size=512, chunk_overlap=50)

ROLE_LABELS = {
    "admin":     "Amministratore",
    "impiegato": "Impiegato",
}


# ── Credenziali: st.secrets (Streamlit Cloud) oppure variabili d'ambiente ─────

def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ[key]


# ── Client Supabase (cached per ciclo di vita dell'app) ───────────────────────

@st.cache_resource
def get_supabase() -> Client:
    return create_client(_secret("SUPABASE_URL"), _secret("SUPABASE_SERVICE_KEY"))


# ── Configurazione LlamaIndex ─────────────────────────────────────────────────

def configure_settings() -> None:
    Settings.llm = Anthropic(
        model="claude-sonnet-4-6",
        system_prompt=SYSTEM_PROMPT,
        api_key=_secret("ANTHROPIC_API_KEY"),
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=_secret("OPENAI_API_KEY"),
    )
    Settings.node_parser = SPLITTER


# ── Indice RAG su Supabase pgvector (cached) ──────────────────────────────────

@st.cache_resource(show_spinner="Caricamento indice in corso…")
def get_index() -> VectorStoreIndex:
    configure_settings()
    vector_store = SupabaseVectorStore(
        postgres_connection_string=_secret("SUPABASE_DB_URL"),
        collection_name="documents",
        dimension=1536,
    )
    return VectorStoreIndex.from_vector_store(vector_store)


def get_query_engine():
    return get_index().as_query_engine(similarity_top_k=5)


# ── Gestione documenti ────────────────────────────────────────────────────────

def _bytes_to_documents(pdf_bytes: bytes, filename: str,
                         categoria: str = "", prodotto: str = "",
                         tipologia: str = "") -> list[Document]:
    """Estrae testo pagina per pagina da bytes PDF con PyMuPDF."""
    docs = []
    basename = Path(filename).name
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / basename
        pdf_path.write_bytes(pdf_bytes)
        fitz_doc = fitz.open(str(pdf_path))
        for page_num in range(len(fitz_doc)):
            text = fitz_doc[page_num].get_text()
            if text.strip():
                docs.append(Document(
                    text=text,
                    metadata={
                        "file_name":  filename,
                        "page_label": str(page_num + 1),
                        "categoria":  categoria,
                        "prodotto":   prodotto,
                        "tipologia":  tipologia,
                    },
                ))
        fitz_doc.close()
    return docs


def add_pdf_to_index(pdf_bytes: bytes, filename: str, categoria: str = "",
                     prodotto: str = "", tipologia: str = "") -> int:
    """Aggiunge un PDF all'indice pgvector in modo incrementale."""
    docs = _bytes_to_documents(pdf_bytes, filename, categoria, prodotto, tipologia)
    if not docs:
        return 0
    nodes = SPLITTER.get_nodes_from_documents(docs)
    get_index().insert_nodes(nodes)
    return len(docs)


def remove_pdf_from_index(filename: str) -> int:
    """Cancella da vecs.documents tutti i chunk con file_name corrispondente."""
    engine = sa.create_engine(_secret("SUPABASE_DB_URL"))
    try:
        with engine.begin() as conn:
            result = conn.execute(
                sa.text(
                    "DELETE FROM vecs.documents "
                    "WHERE metadata->>'file_name' = :fname"
                ),
                {"fname": filename},
            )
            return result.rowcount
    finally:
        engine.dispose()


def _fetch_signed_urls(paths: list[str], expires_in: int = 3600) -> dict[str, str]:
    """Genera signed URL per una lista di bucket paths in una sola chiamata API."""
    if not paths:
        return {}
    try:
        items = get_supabase().storage.from_(BUCKET).create_signed_urls(paths, expires_in) or []
        return {
            item.get("path", ""): item.get("signedURL") or item.get("signed_url") or "#"
            for item in items
            if item.get("path")
        }
    except Exception:
        # fallback per bucket pubblici o versioni diverse di supabase-py
        return {
            p: get_supabase().storage.from_(BUCKET).get_public_url(p)
            for p in paths
        }


def _list_all_pdfs() -> dict[str, list[dict]]:
    """Lista PDF nel bucket suddivisi per categoria.

    Chiavi del dict: etichette di CATEGORIES + "__uncat__" per i file senza prefisso.
    Ogni voce include il campo aggiuntivo "bucket_path" con il path completo nel bucket.
    """
    sb = get_supabase()
    result: dict[str, list[dict]] = {}

    for label, prefix in CATEGORIES.items():
        raw = sb.storage.from_(BUCKET).list(prefix) or []
        files = []
        for f in raw:
            if f["name"].lower().endswith(".pdf"):
                files.append({**f, "bucket_path": f"{prefix}/{f['name']}"})
        result[label] = sorted(files, key=lambda f: f.get("created_at") or "", reverse=True)

    # File alla radice del bucket (caricati prima della categorizzazione)
    raw_root = sb.storage.from_(BUCKET).list("") or []
    uncat = []
    for f in raw_root:
        if f["name"].lower().endswith(".pdf"):
            uncat.append({**f, "bucket_path": f["name"]})
    result["__uncat__"] = sorted(uncat, key=lambda f: f.get("created_at") or "", reverse=True)

    return result


# ── Registro documenti (document_registry) ───────────────────────────────────

def _get_prodotti(ramo: str) -> list[str]:
    """Prodotti distinti già presenti nel registro per il ramo dato, ordinati."""
    resp = (
        get_supabase()
        .table("document_registry")
        .select("prodotto")
        .eq("ramo", ramo)
        .execute()
    )
    return sorted({r["prodotto"] for r in (resp.data or []) if r.get("prodotto")})


def _list_registry_docs() -> dict[str, dict[str, list[dict]]]:
    """Tutti i documenti dal registro, raggruppati per ramo → prodotto.

    Dentro ogni prodotto: ordinati per tipologia (SI > Circ > Norm) poi data desc.
    """
    resp = get_supabase().table("document_registry").select("*").execute()
    docs = resp.data or []

    _tip_ord = {t: i for i, t in enumerate(TIPOLOGIE)}

    def _sort_key(d: dict):
        ts = d.get("data_caricamento") or d.get("created_at") or ""
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            t = 0.0
        return (_tip_ord.get(d.get("tipologia", ""), 99), -t)

    docs.sort(key=_sort_key)
    result: dict[str, dict[str, list[dict]]] = {}
    for doc in docs:
        result.setdefault(doc.get("ramo", ""), {}).setdefault(
            doc.get("prodotto", ""), []
        ).append(doc)
    return result


def _insert_registry(file_path: str, ramo: str, prodotto: str,
                     tipologia: str, edizione: str = "") -> None:
    row: dict = {
        "file_path": file_path,
        "ramo":      ramo,
        "prodotto":  prodotto,
        "tipologia": tipologia,
    }
    if edizione:
        row["edizione"] = edizione
    get_supabase().table("document_registry").insert(row).execute()


def _delete_from_registry(file_path: str) -> None:
    get_supabase().table("document_registry").delete().eq("file_path", file_path).execute()


# ── Generazione report di aggiornamento ──────────────────────────────────────

def _extract_pdf_text(pdf_bytes: bytes, max_chars: int = 80_000) -> str:
    """Estrae il testo completo da PDF bytes usando PyMuPDF, troncato a max_chars."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "doc.pdf"
        p.write_bytes(pdf_bytes)
        doc = fitz.open(str(p))
        text = "\n".join(doc[i].get_text() for i in range(len(doc)))
        doc.close()
    return text[:max_chars]


def _find_previous_si(prodotto: str, exclude_file_path: str) -> dict | None:
    """Restituisce il Set Informativo più recente per il prodotto, escluso il file appena caricato."""
    resp = (
        get_supabase()
        .table("document_registry")
        .select("*")
        .eq("tipologia", "Set Informativo")
        .eq("prodotto", prodotto)
        .neq("file_path", exclude_file_path)
        .order("data_caricamento", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def _update_registry_summary(
    file_path: str,
    change_summary: dict,
    previous_version_id: int | None = None,
) -> None:
    """Aggiorna change_summary (e opzionalmente previous_version_id) nel registro."""
    updates: dict = {"change_summary": change_summary}
    if previous_version_id is not None:
        updates["previous_version_id"] = previous_version_id
    get_supabase().table("document_registry").update(updates).eq("file_path", file_path).execute()


def generate_change_report(
    pdf_bytes: bytes,
    new_file_path: str,
    ramo: str,
    prodotto: str,
    tipologia: str,
) -> dict:
    """Genera un report di aggiornamento per il documento appena caricato.

    Caso A — confronto edizioni: tipologia == "Set Informativo" con precedente esistente.
    Caso B — riassunto singolo: tutti gli altri casi.

    Salva il risultato in change_summary (e previous_version_id se Caso A).
    Solleva eccezione se la generazione fallisce — il chiamante gestisce il fallback.
    """
    new_text  = _extract_pdf_text(pdf_bytes, max_chars=80_000)
    prev_doc  = None
    prev_text = None

    if tipologia == "Set Informativo":
        prev_doc = _find_previous_si(prodotto, new_file_path)
        if prev_doc:
            prev_bytes = get_supabase().storage.from_(BUCKET).download(prev_doc["file_path"])
            prev_text  = _extract_pdf_text(prev_bytes, max_chars=60_000)

    if prev_text is not None:
        # ── Caso A: confronto tra edizioni ────────────────────────────────
        prompt = (
            "Sei un esperto di polizze assicurative italiane. "
            f"Confronta i due Set Informativi del prodotto «{prodotto}» (Ramo: {ramo}). "
            "Identifica con precisione le differenze tra l'edizione precedente e quella nuova. "
            "Non inventare modifiche non presenti nel testo. "
            "Rispondi SOLO con un oggetto JSON valido, senza testo aggiuntivo né backtick markdown, "
            "rispettando esattamente questo schema:\n"
            '{"tipo_report":"confronto","sintesi":"1-2 frasi sul cambiamento principale",'
            '"modifiche":[{"area":"es. Esclusioni RC","descrizione":"cosa è cambiato","impatto":"alto|medio|basso"}],'
            '"novita":["nuove coperture o clausole introdotte"],'
            '"rimozioni":["coperture o clausole rimosse"]}\n\n'
            f"--- EDIZIONE PRECEDENTE ---\n{prev_text[:60_000]}\n\n"
            f"--- NUOVA EDIZIONE ---\n{new_text[:60_000]}"
        )
    else:
        # ── Caso B: riassunto singolo ─────────────────────────────────────
        prompt = (
            "Sei un esperto di polizze assicurative italiane. "
            f"Analizza il seguente documento: tipologia «{tipologia}», "
            f"prodotto «{prodotto}», ramo «{ramo}». "
            "Sii preciso e non inventare informazioni non presenti nel testo. "
            "Rispondi SOLO con un oggetto JSON valido, senza testo aggiuntivo né backtick markdown, "
            "rispettando esattamente questo schema:\n"
            '{"tipo_report":"riassunto","sintesi":"1-2 frasi","punti_chiave":["punti salienti del documento"]}\n\n'
            f"--- DOCUMENTO ---\n{new_text[:80_000]}"
        )

    client = _anthropic_sdk.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    report = json.loads(raw)

    prev_id = int(prev_doc["id"]) if prev_doc and prev_doc.get("id") else None
    _update_registry_summary(new_file_path, report, prev_id)
    return report


# ── Log domande ───────────────────────────────────────────────────────────────

def log_query(username: str, name: str, role: str,
              question: str, answer: str) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "username":  username,
        "name":      name,
        "role":      role,
        "question":  question,
        "answer":    answer,
    }
    with open(LOGS_DIR / "queries.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_query_log(n: int = 30) -> list[dict]:
    log_file = LOGS_DIR / "queries.jsonl"
    if not log_file.exists():
        return []
    entries = []
    with open(log_file, encoding="utf-8") as fh:
        for line in fh:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return list(reversed(entries[-n:]))


# ── Utilità UI ────────────────────────────────────────────────────────────────

def _pdf_row(file_info: dict, bucket_path: str, url: str,
             can_delete: bool = False) -> None:
    """Riga documento: link signed Supabase + (opzionale) elimina.

    bucket_path è il path completo nel bucket (es. "auto/polizza.pdf" o "polizza.pdf").
    url è la signed URL pre-generata da _fetch_signed_urls.
    """
    display_name = Path(bucket_path).name

    created_at = file_info.get("created_at", "")
    try:
        dt        = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        upload_dt = dt.strftime("%d/%m/%y %H:%M")
    except Exception:
        upload_dt = "—"

    if can_delete:
        col_info, col_btn = st.columns([5, 1])
    else:
        col_info = st.container()
        col_btn  = None

    with col_info:
        st.markdown(
            f'<a href="{url}" target="_blank" '
            'style="font-size:13px;line-height:1.35;word-break:break-word;'
            'overflow-wrap:anywhere;color:#1f77b4;text-decoration:none;">'
            f'📄 {display_name}</a>'
            f'<div style="font-size:11px;color:#999;margin-top:2px;">📅 {upload_dt}</div>',
            unsafe_allow_html=True,
        )

    if can_delete and col_btn:
        with col_btn:
            if st.button("🗑️", key=f"del_{bucket_path}",
                         help=f"Rimuovi {display_name}", use_container_width=True):
                st.session_state.confirm_delete = bucket_path
                st.rerun()

    if st.session_state.get("confirm_delete") == bucket_path:
        st.warning(f"Eliminare **{Path(bucket_path).stem}**?")
        c_yes, c_no = st.columns(2)
        with c_yes:
            if st.button("✓ Sì", key=f"yes_{bucket_path}", use_container_width=True):
                with st.spinner("Rimozione…"):
                    get_supabase().storage.from_(BUCKET).remove([bucket_path])
                    remove_pdf_from_index(bucket_path)
                st.session_state.confirm_delete = None
                st.toast(f"«{display_name}» eliminato", icon="🗑️")
                st.rerun()
        with c_no:
            if st.button("✗ No", key=f"no_{bucket_path}", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()

    st.markdown(
        '<hr style="margin:4px 0;border:none;border-top:1px solid #e0e0e0;">',
        unsafe_allow_html=True,
    )


def _registry_doc_row(doc: dict, url: str, can_delete: bool = False) -> None:
    """Riga documento da document_registry: badge tipologia, edizione, data, elimina."""
    bucket_path  = doc.get("file_path", "")
    display_name = Path(bucket_path).name
    tipologia    = doc.get("tipologia", "")
    edizione     = (doc.get("edizione") or "").strip()
    ts           = doc.get("data_caricamento") or doc.get("created_at") or ""
    try:
        upload_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%d/%m/%y %H:%M")
    except Exception:
        upload_dt = "—"

    badge_color = TIPOLOGIA_BADGE_COLOR.get(tipologia, "#888")
    badge_html  = (
        f'<span style="background:{badge_color};color:#fff;font-size:10px;'
        f'padding:1px 6px;border-radius:3px;font-weight:600;white-space:nowrap">'
        f'{tipologia}</span>'
    )
    ed_html = (
        f' <span style="color:#999;font-size:11px">{edizione}</span>'
        if edizione else ""
    )

    if can_delete:
        col_info, col_btn = st.columns([5, 1])
    else:
        col_info = st.container()
        col_btn  = None

    with col_info:
        st.markdown(
            f'<a href="{url}" target="_blank" '
            'style="font-size:13px;line-height:1.35;word-break:break-word;'
            'overflow-wrap:anywhere;color:#1f77b4;text-decoration:none;">'
            f'📄 {display_name}</a><br>'
            f'{badge_html}{ed_html}'
            f'<div style="font-size:11px;color:#999;margin-top:2px;">📅 {upload_dt}</div>',
            unsafe_allow_html=True,
        )

    if can_delete and col_btn:
        with col_btn:
            if st.button("🗑️", key=f"del_{bucket_path}",
                         help=f"Rimuovi {display_name}", use_container_width=True):
                st.session_state.confirm_delete = bucket_path
                st.rerun()

    if st.session_state.get("confirm_delete") == bucket_path:
        st.warning(f"Eliminare **{Path(bucket_path).stem}**?")
        c_yes, c_no = st.columns(2)
        with c_yes:
            if st.button("✓ Sì", key=f"yes_{bucket_path}", use_container_width=True):
                with st.spinner("Rimozione…"):
                    get_supabase().storage.from_(BUCKET).remove([bucket_path])
                    remove_pdf_from_index(bucket_path)
                    _delete_from_registry(bucket_path)
                st.session_state.confirm_delete = None
                st.toast(f"«{display_name}» eliminato", icon="🗑️")
                st.rerun()
        with c_no:
            if st.button("✗ No", key=f"no_{bucket_path}", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()

    st.markdown(
        '<hr style="margin:4px 0;border:none;border-top:1px solid #e0e0e0;">',
        unsafe_allow_html=True,
    )


@st.dialog("📤 Carica documento")
def _upload_dialog() -> None:
    """Modale per caricare un nuovo PDF con Ramo, Prodotto, Tipologia ed Edizione.

    Usa due fasi:
    - Fase 1 (form): l'utente compila i campi e clicca "Carica".
      Il click salva i dati in session_state, incrementa uploader_key e chiama
      st.rerun(). Il dialog rimane aperto perché _show_upload_dialog è ancora True.
    - Fase 2 (upload): mostra UI di caricamento, esegue upload+indicizzazione,
      poi azzera _show_upload_dialog e chiama st.rerun() che chiude la modale.
    """

    # ── Fase 2: upload in corso ────────────────────────────────────────────
    if st.session_state.get("_dlg_uploading"):
        params = st.session_state["_upload_params"]
        st.info(f"Caricamento di **{params['name']}** in corso…")

        with st.spinner("Caricamento e indicizzazione…"):
            _prefix      = CATEGORIES[params["ramo"]]
            _bucket_path = f"{_prefix}/{params['name']}"
            get_supabase().storage.from_(BUCKET).upload(
                path=_bucket_path,
                file=params["bytes"],
                file_options={"content-type": "application/pdf", "upsert": "true"},
            )
            _insert_registry(
                _bucket_path, params["ramo"], params["prodotto"],
                params["tipologia"], params["edizione"],
            )
            n = add_pdf_to_index(
                params["bytes"], _bucket_path,
                categoria=_prefix,
                prodotto=params["prodotto"],
                tipologia=params["tipologia"],
            )

        _report_ok = True
        with st.spinner("Generazione report di aggiornamento…"):
            try:
                generate_change_report(
                    params["bytes"], _bucket_path,
                    params["ramo"], params["prodotto"], params["tipologia"],
                )
            except Exception as _e:
                import traceback
                traceback.print_exc()   # visibile nei log di Streamlit
                _report_ok = False

        st.session_state.pop("_dlg_uploading", None)
        st.session_state.pop("_upload_params", None)
        st.session_state.pop("_show_upload_dialog", None)   # chiude la modale al prossimo render

        if _report_ok:
            st.toast(
                f"«{params['name']}» aggiunto ({params['ramo']} · {params['prodotto']})"
                f" — {n} pagine indicizzate",
                icon="✅",
            )
        else:
            st.toast(
                f"«{params['name']}» caricato. Report non generato, riprovare più tardi.",
                icon="⚠️",
            )
        st.rerun()
        return

    # ── Fase 1: form ───────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Seleziona PDF",
        type="pdf",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    _ramo_label = st.selectbox("Ramo *", options=list(CATEGORIES.keys()), key="dlg_ramo")

    _prodotti_esistenti = _get_prodotti(_ramo_label)
    _prodotto_sel = st.selectbox(
        "Prodotto *",
        options=_prodotti_esistenti + ["➕ Nuovo prodotto"],
        key=f"dlg_prodotto_{_ramo_label}",
    )
    if _prodotto_sel == "➕ Nuovo prodotto":
        _prodotto_finale = st.text_input(
            "Nome nuovo prodotto", placeholder="es. Sara Più Auto", key="dlg_prodotto_nuovo"
        ).strip()
    else:
        _prodotto_finale = _prodotto_sel

    _tipologia_sel = st.selectbox(
        "Tipologia *", options=["— scegli —"] + TIPOLOGIE, key="dlg_tipologia"
    )
    _tipologia_ok = _tipologia_sel in TIPOLOGIE

    _edizione = st.text_input(
        "Edizione (opzionale)", placeholder="es. ed. 4/2026", key="dlg_edizione"
    ).strip()

    st.divider()

    _campi_ok = bool(uploaded and _prodotto_finale and _tipologia_ok)
    if _prodotto_sel == "➕ Nuovo prodotto" and not _prodotto_finale:
        st.caption("⚠️ Inserisci il nome del prodotto.")
    if not _tipologia_ok:
        st.caption("⚠️ Seleziona la tipologia.")

    if st.button("Carica", use_container_width=True, type="primary", disabled=not _campi_ok):
        _prefix      = CATEGORIES[_ramo_label]
        _bucket_path = f"{_prefix}/{uploaded.name}"
        _dup = (
            get_supabase()
            .table("document_registry")
            .select("file_path")
            .eq("file_path", _bucket_path)
            .execute()
        )
        if _dup.data:
            st.warning(f"**{uploaded.name}** è già registrato in «{_ramo_label}».")
        else:
            st.session_state["_dlg_uploading"] = True
            st.session_state["_upload_params"] = {
                "bytes":     uploaded.getvalue(),
                "name":      uploaded.name,
                "ramo":      _ramo_label,
                "prodotto":  _prodotto_finale,
                "tipologia": _tipologia_sel,
                "edizione":  _edizione,
            }
            st.session_state.uploader_key += 1  # svuota il file uploader al prossimo render
            st.rerun()  # _show_upload_dialog è ancora True → dialog rimane aperto in fase 2


def highlight_citations(text: str) -> str:
    pattern = r"(\*?\(?\*?[\w][\w\s\.\-\/]+\.pdf\*?[,\s]*(?:pag\.|p\.)\s*\d+\*?\)?)"
    repl = (
        r'<span style="background:#fff3cd;padding:2px 6px;'
        r'border-radius:4px;font-size:.9em;">📄 \1</span>'
    )
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


# ── Vista report aggiornamenti ────────────────────────────────────────────────

_IMPACT_COLOR: dict[str, str] = {
    "alto":  "#D32F2F",
    "medio": "#F57C00",
    "basso": "#388E3C",
}


def _render_report_card(doc: dict, url: str, compact: bool = False) -> None:
    """Scheda-report per un documento con change_summary parsato.

    compact=True → solo intestazione + sintesi + nota (senza expander di dettaglio).
    compact=False → aggiunge st.expander con modifiche/punti_chiave + link PDF.
    """
    cs        = doc.get("_report", {})
    prodotto  = doc.get("prodotto", "—")
    tipologia = doc.get("tipologia", "—")
    edizione  = (doc.get("edizione") or "").strip()
    ts        = doc.get("data_caricamento") or doc.get("created_at") or ""
    try:
        data_str = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%d/%m/%Y")
    except Exception:
        data_str = "—"

    sintesi     = cs.get("sintesi", "")
    tipo_report = cs.get("tipo_report", "")
    badge_color = TIPOLOGIA_BADGE_COLOR.get(tipologia, "#888")

    badge_html = (
        f'<span style="background:{badge_color};color:#fff;font-size:10px;'
        f'padding:2px 7px;border-radius:3px;font-weight:600;white-space:nowrap">'
        f'{tipologia}</span>'
    )
    ed_html = (
        f'<span style="color:#999;font-size:12px">{edizione}</span>'
        if edizione else ""
    )

    st.markdown(
        f'<div class="report-card">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px">'
        f'<span style="font-weight:600;font-size:14px;color:#1A1A2E">{prodotto}</span>'
        f'{badge_html}{ed_html}'
        f'<span style="margin-left:auto;color:#aaa;font-size:11px">{data_str}</span>'
        f'</div>'
        f'<div style="font-size:13px;color:#444;line-height:1.5">{sintesi}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not compact:
        with st.expander("Dettaglio"):
            if tipo_report == "confronto":
                modifiche = cs.get("modifiche", [])
                if modifiche:
                    st.markdown("**Modifiche**")
                    for m in modifiche:
                        impatto   = (m.get("impatto") or "").lower()
                        imp_color = _IMPACT_COLOR.get(impatto, "#888")
                        imp_badge = (
                            f'<span style="background:{imp_color};color:#fff;font-size:10px;'
                            f'padding:1px 6px;border-radius:3px;font-weight:600">{impatto}</span>'
                        )
                        st.markdown(
                            f'{imp_badge} &nbsp; **{m.get("area", "")}** — {m.get("descrizione", "")}',
                            unsafe_allow_html=True,
                        )
                novita = cs.get("novita", [])
                if novita:
                    st.markdown("**Novità introdotte**")
                    for n in novita:
                        st.markdown(f"- {n}")
                rimozioni = cs.get("rimozioni", [])
                if rimozioni:
                    st.markdown("**Rimozioni**")
                    for r in rimozioni:
                        st.markdown(f"- {r}")
            elif tipo_report == "riassunto":
                punti = cs.get("punti_chiave", [])
                if punti:
                    st.markdown("**Punti chiave**")
                    for p in punti:
                        st.markdown(f"- {p}")

            st.divider()
            st.caption("⚠️ Sintesi generata automaticamente — verificare sul documento originale")
            if url and url != "#":
                st.markdown(f"[📄 Apri documento]({url})")
    else:
        st.markdown(
            '<div style="font-size:11px;color:#aaa;margin-top:2px">'
            '⚠️ Sintesi generata automaticamente — verificare sul documento originale'
            f'{"  ·  " + f"""<a href="{url}" target="_blank">📄 Apri documento</a>""" if url and url != "#" else ""}'
            '</div>',
            unsafe_allow_html=True,
        )


def _render_aggiornamenti_tab() -> None:
    """Contenuto del tab Aggiornamenti: novità recenti + vista per ramo."""
    resp = (
        get_supabase()
        .table("document_registry")
        .select("*")
        .order("data_caricamento", desc=True)
        .execute()
    )

    valid_docs: list[dict] = []
    for doc in resp.data or []:
        cs = doc.get("change_summary")
        if cs is None:
            continue
        try:
            if isinstance(cs, str):
                cs = json.loads(cs)
            if isinstance(cs, dict) and cs.get("tipo_report"):
                valid_docs.append({**doc, "_report": cs})
        except Exception:
            pass

    if not valid_docs:
        st.info(
            "Nessun aggiornamento disponibile. "
            "I report vengono generati automaticamente al caricamento di nuovi documenti."
        )
        return

    urls = _fetch_signed_urls([d["file_path"] for d in valid_docs])

    # ── Novità recenti (ultimi 5) ─────────────────────────────────────────────
    st.markdown("### Novità recenti")
    for doc in valid_docs[:5]:
        _render_report_card(doc, url=urls.get(doc["file_path"], "#"), compact=True)

    st.divider()

    # ── Vista per ramo ────────────────────────────────────────────────────────
    st.markdown("### Tutti gli aggiornamenti")

    by_ramo: dict[str, dict[str, list[dict]]] = {}
    for doc in valid_docs:
        by_ramo.setdefault(doc.get("ramo", ""), {}).setdefault(
            doc.get("prodotto", ""), []
        ).append(doc)

    for ramo_label in CATEGORIES:
        prodotti = by_ramo.get(ramo_label, {})
        count    = sum(len(v) for v in prodotti.values())
        icon     = CATEGORY_ICONS[ramo_label]

        with st.expander(f"{icon} {ramo_label}  ({count})", expanded=False):
            if not prodotti:
                st.caption("Nessun report disponibile per questo ramo.")
                continue
            for prodotto in sorted(prodotti.keys()):
                st.markdown(f"**{prodotto}**")
                for doc in prodotti[prodotto]:
                    _render_report_card(
                        doc, url=urls.get(doc["file_path"], "#"), compact=False
                    )


# ── Configurazione pagina ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Oracolo delle Polizze",
    page_icon=_PILImage.open(Path(__file__).parent.parent / "static" / "logo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Nascondi elementi Streamlit ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header: nascosto ma overflow visibile per il pulsante espandi sidebar */
    header {visibility: hidden; height: 0; overflow: visible;}
    /* stToolbar: solo invisible, NON display:none (altrimenti stExpandSidebarButton scompare) */
    [data-testid="stToolbar"] {visibility: hidden;}
    /* Pulsante espandi sidebar: ripristinato e ancorato fixed in alto a sinistra */
    [data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        display: flex !important;
        position: fixed !important;
        top: 0.4rem !important;
        left: 0.6rem !important;
        z-index: 999999 !important;
    }
    .stDeployButton {display: none;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    .stAppDeployButton {display: none;}
    .viewerBadge_container__1QSob {display: none;}
    .styles_viewerBadge__1yB5_ {display: none;}
    a[href*="streamlit.io"] {display: none !important;}

    /* ── Titolo principale ── */
    h1 {
        color: #8B2061 !important;
        font-weight: 700;
    }

    /* ── Pulsanti primari (Chiedi, Upload, Accedi) ── */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background-color: #F47920 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {
        background-color: #d96a10 !important;
        color: #FFFFFF !important;
    }

    /* ── Area risposta (solo blocchi esplicitamente marcati) ── */
    .sara-risposta {
        background: #FFFFFF;
        border-left: 4px solid #8B2061;
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 8px;
    }

    /* ── Tabelle nelle risposte ── */
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
    }
    thead tr th {
        background-color: #EDD6E8;
        color: #1A1A2E;
        padding: 8px 12px;
        text-align: left;
        border-bottom: 2px solid #8B2061;
    }
    tbody tr:nth-child(even) {
        background-color: #F7F7F9;
    }
    tbody tr:nth-child(odd) {
        background-color: #FFFFFF;
    }
    tbody td {
        padding: 7px 12px;
        border-bottom: 1px solid #E8E8EC;
    }

    /* ── Report card (tab Aggiornamenti) ── */
    .report-card {
        background: #fff;
        border-left: 4px solid #8B2061;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px 10px 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Login gate ────────────────────────────────────────────────────────────────

if not st.session_state.get("authenticated"):

    # ── Fase intermedia: credenziali ok, rendering app in corso ──────────────
    # Il browser riceve questa schermata e la mostra MENTRE il prossimo rerun
    # (pesante: query sidebar, indice RAG) è in esecuzione lato server.
    if st.session_state.get("_logging_in"):
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'justify-content:center;min-height:65vh;gap:20px;text-align:center">'
            f'<img src="{_LOGO_SRC}" style="width:72px;height:72px">'
            f'<div style="font-size:1.6rem;font-weight:700;color:#8B2061">'
            f'Oracolo delle Polizze</div>'
            f'<div style="color:#999;font-size:14px">Caricamento in corso…</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.spinner(""):
            pass
        st.session_state["authenticated"]  = True
        st.session_state["_open_sidebar"]  = True
        st.session_state.pop("_logging_in", None)
        st.rerun()

    # ── Form di login ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="text-align:center;padding:16px 0 8px 0">'
        f'<img src="{_LOGO_SRC}" style="width:64px;height:64px;display:block;margin:0 auto 14px;">'
        '<div style="font-size:2rem;font-weight:700;color:#8B2061;margin-bottom:6px">Oracolo delle Polizze</div>'
        '<div style="color:#888;font-size:15px">Interroga i tuoi documenti assicurativi</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    _, col_login, _ = st.columns([1, 1, 1])
    with col_login:
        with st.form("login_form"):
            _uname_input = st.text_input("Username")
            _pwd_input   = st.text_input("Password", type="password")
            _submitted   = st.form_submit_button("🔑 Accedi", use_container_width=True)

        if _submitted:
            try:
                _res = (
                    get_supabase()
                    .table("users")
                    .select("username,name,password_hash,role")
                    .eq("username", _uname_input)
                    .limit(1)
                    .execute()
                )
                _row = _res.data[0] if _res.data else None
                if _row and bcrypt.checkpw(
                    _pwd_input.encode("utf-8"),
                    _row["password_hash"].encode("utf-8"),
                ):
                    # Fase 1: salva i dati utente e attiva la schermata di caricamento
                    st.session_state["username"] = _row["username"]
                    st.session_state["name"]     = _row["name"]
                    st.session_state["role"]     = _row.get("role", "impiegato")
                    st.session_state["_logging_in"] = True
                    st.rerun()
                else:
                    st.error("🔑 Username o password non corretti.")
            except Exception:
                st.error("🔑 Username o password non corretti.")
    st.stop()

# ── Utente autenticato ────────────────────────────────────────────────────────

_uname      = st.session_state["username"]
_name       = st.session_state["name"]
_role       = st.session_state["role"]
_role_label = ROLE_LABELS.get(_role, _role)

# ── Forza sidebar aperta al primo render dopo il login ───────────────────────

if st.session_state.pop("_open_sidebar", False):
    _components.html(
        """<script>
        const tryOpen = () => {
            const btn = window.parent.document.querySelector(
                '[data-testid="stExpandSidebarButton"]');
            const sidebar = window.parent.document.querySelector(
                '[data-testid="stSidebar"]');
            if (sidebar && sidebar.getAttribute('aria-expanded') === 'false' && btn) {
                btn.click();
            } else if (!sidebar) {
                setTimeout(tryOpen, 250);
            }
        };
        setTimeout(tryOpen, 400);
        </script>""",
        height=0,
    )

# ── Stato sessione ────────────────────────────────────────────────────────────

_defaults = {
    "history":          [],
    "last_answer":      None,
    "last_question":    None,
    "is_loading":       False,
    "pending_question": None,
    "confirm_delete":   None,
    "uploader_key":     0,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0 2px 0">'
        f'<img src="{_LOGO_SRC}" style="width:36px;height:36px;">'
        '<span style="font-size:1.4rem;font-weight:700;color:#8B2061">Oracolo</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Gestione Documenti ────────────────────────────────────────────────────
    st.subheader("📂 Gestione Documenti")

    # — Upload (admin e impiegato) —
    if _role in ("admin", "impiegato"):
        if st.button("📤 Carica documento", use_container_width=True, type="primary"):
            st.session_state["_show_upload_dialog"] = True
        if st.session_state.get("_show_upload_dialog", False):
            _upload_dialog()
        st.divider()

    # — Lista documenti —
    _search = st.text_input(
        "Cerca documento",
        placeholder="🔍 Filtra per nome…",
        label_visibility="collapsed",
    )

    _reg_docs    = _list_registry_docs()
    _reg_paths   = [
        d["file_path"]
        for ramo_dict in _reg_docs.values()
        for docs_list in ramo_dict.values()
        for d in docs_list
    ]
    _signed_urls = _fetch_signed_urls(_reg_paths)

    for _ramo_label, _prefix in CATEGORIES.items():
        _ramo_dict = _reg_docs.get(_ramo_label, {})

        # Applica filtro di ricerca per prodotto
        if _search:
            _ramo_dict = {
                prod: [d for d in docs
                       if _search.lower() in Path(d["file_path"]).name.lower()]
                for prod, docs in _ramo_dict.items()
            }
            _ramo_dict = {k: v for k, v in _ramo_dict.items() if v}

        _count = sum(len(v) for v in _ramo_dict.values())
        _icon  = CATEGORY_ICONS[_ramo_label]

        with st.expander(f"{_icon} {_ramo_label}  ({_count})", expanded=False):
            if _ramo_dict:
                for _prodotto in sorted(_ramo_dict.keys()):
                    st.markdown(f"**{_prodotto}**")
                    for _doc in _ramo_dict[_prodotto]:
                        _registry_doc_row(
                            _doc,
                            url=_signed_urls.get(_doc["file_path"], "#"),
                            can_delete=(_role == "admin"),
                        )
            else:
                st.caption("Nessun documento." if not _search else "Nessun risultato.")

    # — File nel bucket senza voce nel registro —
    _all_bucket = _list_all_pdfs()
    _reg_path_set = set(_reg_paths)
    _uncat_files = [
        fi
        for bucket_list in _all_bucket.values()
        for fi in bucket_list
        if fi["bucket_path"] not in _reg_path_set
    ]
    if _search:
        _uncat_files = [f for f in _uncat_files
                        if _search.lower() in f["name"].lower()]
    if _uncat_files:
        _uncat_urls = _fetch_signed_urls([f["bucket_path"] for f in _uncat_files])
        with st.expander(f"⚠️ Da categorizzare  ({len(_uncat_files)})", expanded=False):
            for _fi in _uncat_files:
                _pdf_row(
                    _fi,
                    bucket_path=_fi["bucket_path"],
                    url=_uncat_urls.get(_fi["bucket_path"], "#"),
                    can_delete=(_role == "admin"),
                )

    st.divider()

    # — Log domande (solo admin) —
    if _role == "admin":
        with st.expander("📊 Log domande recenti"):
            _log_entries = load_query_log(20)
            if _log_entries:
                for _e in _log_entries:
                    _dt = datetime.fromisoformat(_e["timestamp"]).strftime("%d/%m/%y %H:%M")
                    st.markdown(
                        f'<div style="font-size:12px;color:#555;margin-top:6px">'
                        f'🕐 {_dt} &nbsp;·&nbsp; '
                        f'<b>{_e["name"]}</b> <span style="color:#888">({_e["role"]})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="font-size:12px;margin:2px 0 0 4px">'
                        f'❓ {_e["question"]}</div>',
                        unsafe_allow_html=True,
                    )
                    _ans_preview = _e["answer"][:180] + ("…" if len(_e["answer"]) > 180 else "")
                    st.markdown(
                        f'<div style="font-size:11px;color:#777;margin:2px 0 6px 4px">'
                        f'💬 {_ans_preview}</div>'
                        '<hr style="margin:4px 0;border:none;border-top:1px solid #eee">',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Nessuna domanda registrata.")
        st.divider()

    # — Nuova sessione —
    if st.button("🗑️ Nuova sessione", use_container_width=True):
        for _k in ["history", "last_answer", "last_question",
                   "is_loading", "pending_question"]:
            st.session_state.pop(_k, None)
        st.rerun()

    st.divider()
    st.caption("**Modello LLM:** claude-sonnet-4-6")
    st.caption("**Embedding:** text-embedding-3-small")
    st.caption("**Chunk:** 512 token · overlap 50")


# ── Area principale ───────────────────────────────────────────────────────────

_col_title, _col_user = st.columns([5, 3])
with _col_title:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:2px">'
        f'<img src="{_LOGO_SRC}" style="width:44px;height:44px;flex-shrink:0;">'
        '<span style="font-size:2rem;font-weight:700;color:#8B2061;line-height:1.2">Oracolo delle Polizze</span>'
        '</div>'
        '<div style="color:#666;margin-top:8px;font-size:15px;line-height:1.5">'
        'Consulta i Documenti di Sara Assicurazioni.<br>'
        '<span style="font-size:13px;color:#999">'
        'Le risposte si basano esclusivamente sui documenti caricati.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with _col_user:
    st.markdown(
        f'<div style="background:#F7F7F9;border:1px solid #E8E8EC;border-radius:8px;'
        f'padding:12px 16px;text-align:right;margin-top:8px">'
        f'<div style="font-size:14px;font-weight:600;color:#1A1A2E">👤 {_name}</div>'
        f'<div style="font-size:12px;color:#8B2061;font-weight:500;margin-top:3px">'
        f'{_role_label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("🚪 Esci", key="logout_btn", use_container_width=True):
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        st.rerun()

st.markdown('<div style="margin:12px 0 4px 0"></div>', unsafe_allow_html=True)
st.divider()

_tab_oracolo, _tab_aggiornamenti = st.tabs(["Oracolo", "Aggiornamenti"])

# ── Tab: Oracolo ──────────────────────────────────────────────────────────────

with _tab_oracolo:

    with st.form("query_form", clear_on_submit=False):
        domanda = st.text_area(
            "La tua domanda",
            placeholder="Es: Quali sono i massimali per la garanzia infortuni conducente?",
            height=110,
            disabled=st.session_state.is_loading,
        )
        submitted = st.form_submit_button(
            "🔍 Chiedi",
            use_container_width=True,
            disabled=st.session_state.is_loading,
        )

    if st.session_state.is_loading:
        st.info("⏳ L'Oracolo sta consultando i documenti… un momento.")

    # Fase 1 — salva domanda e avvia il ciclo di caricamento
    if submitted and domanda.strip() and not st.session_state.is_loading:
        st.session_state.pending_question = domanda.strip()
        st.session_state.is_loading = True
        st.rerun()
    elif submitted and not domanda.strip():
        st.toast("Scrivi prima la tua domanda 😊", icon="✏️")

    # Fase 2 — elaborazione (pulsante già disabilitato)
    if st.session_state.is_loading and st.session_state.pending_question:
        with st.spinner("L'Oracolo sta consultando i documenti…"):
            risposta = str(get_query_engine().query(st.session_state.pending_question))

        log_query(_uname, _name, _role,
                  st.session_state.pending_question, risposta)

        st.session_state.last_question    = st.session_state.pending_question
        st.session_state.last_answer      = risposta
        st.session_state.history.insert(0, {
            "domanda":  st.session_state.pending_question,
            "risposta": risposta,
        })
        st.session_state.history          = st.session_state.history[:5]
        st.session_state.is_loading       = False
        st.session_state.pending_question = None
        st.rerun()

    # ── Risposta corrente ─────────────────────────────────────────────────────

    if st.session_state.last_answer:
        st.markdown(
            f'<h3 style="color:#8B2061;margin-top:28px;margin-bottom:10px">'
            f'❓ {st.session_state.last_question}</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sara-risposta">{highlight_citations(st.session_state.last_answer)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="margin-bottom:20px"></div>', unsafe_allow_html=True)
        st.divider()

    # ── Cronologia ────────────────────────────────────────────────────────────

    _previous = st.session_state.history[1:] if st.session_state.history else []

    if _previous:
        st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
        st.subheader("🕐 Ultime domande")
        for _item in _previous:
            with st.expander(f"💬 {_item['domanda']}"):
                st.markdown(
                    f'<div class="sara-risposta">{highlight_citations(_item["risposta"])}</div>',
                    unsafe_allow_html=True,
                )

# ── Tab: Aggiornamenti ────────────────────────────────────────────────────────

with _tab_aggiornamenti:
    _render_aggiornamenti_tab()
