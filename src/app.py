import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pymupdf as fitz
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic

# ── Percorsi ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
DOCUMENTS_DIR = BASE_DIR / "documents"
STORAGE_DIR   = BASE_DIR / "storage"
STATIC_DIR    = Path(__file__).parent / "static"   # src/static/ → /app/static/

SYSTEM_PROMPT = (
    "Sei l'Oracolo delle Polizze. Rispondi SOLO basandoti sui documenti forniti. "
    "Cita sempre il nome del documento e la pagina. "
    "Se non trovi l'informazione, dì esplicitamente che non è presente nei documenti. "
    "Non inventare mai coperture o esclusioni."
)

SPLITTER = SentenceSplitter(chunk_size=512, chunk_overlap=50)


# ── Configurazione LlamaIndex ─────────────────────────────────────────────────

def configure_settings() -> None:
    Settings.llm = Anthropic(
        model="claude-sonnet-4-6",
        system_prompt=SYSTEM_PROMPT,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.node_parser = SPLITTER


# ── Caricamento PDF ───────────────────────────────────────────────────────────

def _pdf_to_documents(pdf_path: Path) -> list[Document]:
    """Estrae testo pagina per pagina con PyMuPDF."""
    docs = []
    fitz_doc = fitz.open(str(pdf_path))
    for page_num in range(len(fitz_doc)):
        text = fitz_doc[page_num].get_text()
        if text.strip():
            docs.append(Document(
                text=text,
                metadata={
                    "file_name":  pdf_path.name,
                    "page_label": str(page_num + 1),
                },
            ))
    fitz_doc.close()
    return docs


def load_pdfs_with_pymupdf(docs_dir: Path) -> list[Document]:
    """Carica tutti i PDF dalla cartella."""
    all_docs: list[Document] = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        all_docs.extend(_pdf_to_documents(pdf_path))
    return all_docs


# ── Indice RAG (cached) ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Caricamento indice in corso…")
def get_index() -> VectorStoreIndex:
    """
    Carica l'indice dallo storage se esiste, altrimenti lo costruisce da zero.
    L'oggetto restituito è mutabile: add/remove agiscono su di esso in-place.
    """
    configure_settings()
    if STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir()):
        sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        return load_index_from_storage(sc)

    documents = load_pdfs_with_pymupdf(DOCUMENTS_DIR)
    if documents:
        index = VectorStoreIndex.from_documents(documents, transformations=[SPLITTER])
    else:
        index = VectorStoreIndex([])          # indice vuoto, riempito dopo
    STORAGE_DIR.mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    return index


def get_query_engine():
    return get_index().as_query_engine(similarity_top_k=5)


# ── Aggiunta incrementale ─────────────────────────────────────────────────────

def add_pdf_to_index(pdf_path: Path) -> int:
    """
    Aggiunge un PDF all'indice in modo incrementale.
    Restituisce il numero di pagine con testo indicizzate.
    """
    configure_settings()

    # Se lo storage non esisteva, get_index() lo ha appena costruito
    # includendo il file già salvato su disco → basta contare le pagine.
    storage_pre_existed = STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir())
    index = get_index()

    new_docs = _pdf_to_documents(pdf_path)

    if not storage_pre_existed:
        # Indice costruito da zero con tutti i file presenti: il PDF è già dentro.
        return len(new_docs)

    # Indice caricato dallo storage precedente → inserimento incrementale.
    if new_docs:
        nodes = SPLITTER.get_nodes_from_documents(new_docs)
        index.insert_nodes(nodes)
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    return len(new_docs)


# ── Rimozione dal indice ──────────────────────────────────────────────────────

def remove_pdf_from_index(filename: str) -> int:
    """
    Rimuove dall'indice tutti i chunk appartenenti al file indicato.
    Restituisce il numero di ref_doc rimossi.
    """
    index   = get_index()
    docstore = index.storage_context.docstore

    # Raccoglie i ref_doc_id univoci di tutti i chunk del file
    ref_doc_ids: set[str] = {
        node.ref_doc_id
        for node in docstore.docs.values()
        if node.metadata.get("file_name") == filename
        and node.ref_doc_id
    }

    removed = 0
    for ref_id in ref_doc_ids:
        try:
            index.delete_ref_doc(ref_id, delete_from_docstore=True)
            removed += 1
        except Exception:
            pass

    if removed:
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    return removed


# ── Sincronizzazione static/ ──────────────────────────────────────────────────

def sync_static_dir() -> None:
    """Copia tutti i PDF da documents/ in src/static/ (idempotente)."""
    STATIC_DIR.mkdir(exist_ok=True)
    for pdf in DOCUMENTS_DIR.glob("*.pdf"):
        dest = STATIC_DIR / pdf.name
        if not dest.exists():
            shutil.copy2(pdf, dest)


def static_add(pdf_path: Path) -> None:
    """Aggiunge un singolo PDF a src/static/."""
    STATIC_DIR.mkdir(exist_ok=True)
    shutil.copy2(pdf_path, STATIC_DIR / pdf_path.name)


def static_remove(filename: str) -> None:
    """Rimuove un PDF da src/static/ se esiste."""
    target = STATIC_DIR / filename
    if target.exists():
        target.unlink()


# ── Utilità UI ────────────────────────────────────────────────────────────────

def _pdf_link_row(pdf: Path) -> None:
    """
    Mostra il nome del file come link che apre il PDF in una nuova scheda
    tramite lo static file serving di Streamlit (/app/static/<nome>).
    """
    upload_dt = datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%d/%m/%y %H:%M")
    url = f"/app/static/{pdf.name}"
    st.markdown(
        f'<a href="{url}" target="_blank" '
        'style="font-size:13px;line-height:1.35;word-break:break-word;'
        'overflow-wrap:anywhere;color:#1f77b4;text-decoration:none;">'
        f'📄 {pdf.name}</a>'
        f'<div style="font-size:11px;color:#999;margin-top:2px;">📅 {upload_dt}</div>',
        unsafe_allow_html=True,
    )


def highlight_citations(text: str) -> str:
    """Evidenzia le citazioni (nome.pdf, pag. N) con sfondo giallo."""
    pattern = r"(\*?\(?\*?[\w][\w\s\.\-]+\.pdf\*?[,\s]*(?:pag\.|p\.)\s*\d+\*?\)?)"
    repl = (
        r'<span style="background:#fff3cd;padding:2px 6px;'
        r'border-radius:4px;font-size:.9em;">📄 \1</span>'
    )
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


# ── Configurazione pagina ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Oracolo delle Polizze",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sincronizza src/static/ con documents/ all'avvio (idempotente)
sync_static_dir()

# ── Stato sessione ────────────────────────────────────────────────────────────

defaults = {
    "history":          [],
    "last_answer":      None,
    "last_question":    None,
    "is_loading":       False,
    "pending_question": None,
    "confirm_delete":   None,   # nome del file in attesa di conferma
    "last_upload_key":  None,   # (nome + size) dell'ultimo upload processato
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔮 Oracolo delle Polizze")
    st.divider()

    # ── Gestione Documenti ────────────────────────────────────────────────────
    st.subheader("📂 Gestione Documenti")

    # File uploader
    uploaded = st.file_uploader(
        "Carica un nuovo PDF",
        type="pdf",
        help="Il documento viene aggiunto all'indice senza reindicizzare gli altri.",
    )

    if uploaded is not None:
        upload_key = f"{uploaded.name}_{uploaded.size}"
        if upload_key != st.session_state.last_upload_key:
            dest = DOCUMENTS_DIR / uploaded.name
            if dest.exists():
                st.warning(f"**{uploaded.name}** è già presente.")
            else:
                dest.write_bytes(uploaded.getbuffer())
                static_add(dest)
                with st.spinner(f"Indicizzazione di «{uploaded.name}»…"):
                    n = add_pdf_to_index(dest)
                st.session_state.last_upload_key = upload_key
                st.toast(
                    f"«{uploaded.name}» aggiunto — {n} pagine indicizzate",
                    icon="✅",
                )
                st.rerun()
            st.session_state.last_upload_key = upload_key

    st.divider()

    # Lista documenti
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if pdf_files:
        for pdf in pdf_files:
            col_info, col_btn = st.columns([5, 1])
            with col_info:
                _pdf_link_row(pdf)
            with col_btn:
                if st.button(
                    "🗑️",
                    key=f"del_{pdf.name}",
                    help=f"Rimuovi {pdf.name}",
                    use_container_width=True,
                ):
                    st.session_state.confirm_delete = pdf.name
                    st.rerun()

            # Conferma eliminazione (inline, sotto la riga del file)
            if st.session_state.confirm_delete == pdf.name:
                st.warning(f"Eliminare **{pdf.stem}**?")
                c_yes, c_no = st.columns(2)
                with c_yes:
                    if st.button(
                        "✓ Sì", key=f"yes_{pdf.name}", use_container_width=True
                    ):
                        with st.spinner("Rimozione…"):
                            remove_pdf_from_index(pdf.name)
                            pdf.unlink()
                            static_remove(pdf.name)
                        st.session_state.confirm_delete = None
                        st.toast(f"«{pdf.name}» eliminato", icon="🗑️")
                        st.rerun()
                with c_no:
                    if st.button(
                        "✗ No", key=f"no_{pdf.name}", use_container_width=True
                    ):
                        st.session_state.confirm_delete = None
                        st.rerun()

            # Separatore sottile tra file (meno spazio di st.divider / ---)
            st.markdown(
                '<hr style="margin:4px 0;border:none;border-top:1px solid #e0e0e0;">',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Nessun documento ancora. Carica un PDF qui sopra.")

    st.divider()

    # Nuova sessione
    if st.button("🗑️ Nuova sessione", use_container_width=True):
        for k in ["history", "last_answer", "last_question",
                  "is_loading", "pending_question"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    st.caption("**Modello LLM:** claude-sonnet-4-6")
    st.caption("**Embedding:** text-embedding-3-small")
    st.caption("**Chunk:** 512 token · overlap 50")


# ── Area principale ───────────────────────────────────────────────────────────

st.title("Oracolo delle Polizze")
st.caption(
    "Fai una domanda sulle tue polizze assicurative: "
    "rispondo solo in base ai documenti caricati."
)

st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────

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

# Messaggio di caricamento
if st.session_state.is_loading:
    st.info("⏳ L'Oracolo sta consultando i documenti… un momento.", icon="🔮")

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

# ── Risposta corrente ─────────────────────────────────────────────────────────

if st.session_state.last_answer:
    st.subheader(f"❓ {st.session_state.last_question}")
    st.markdown(
        highlight_citations(st.session_state.last_answer),
        unsafe_allow_html=True,
    )
    st.divider()

# ── Cronologia ────────────────────────────────────────────────────────────────

previous = st.session_state.history[1:] if st.session_state.history else []

if previous:
    st.subheader("🕐 Ultime domande")
    for item in previous:
        with st.expander(f"💬 {item['domanda']}"):
            st.markdown(
                highlight_citations(item["risposta"]),
                unsafe_allow_html=True,
            )
