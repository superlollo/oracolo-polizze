import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

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

BUCKET = "cga-documents"

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

def _bytes_to_documents(pdf_bytes: bytes, filename: str) -> list[Document]:
    """Estrae testo pagina per pagina da bytes PDF con PyMuPDF."""
    docs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / filename
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
                    },
                ))
        fitz_doc.close()
    return docs


def add_pdf_to_index(pdf_bytes: bytes, filename: str) -> int:
    """Aggiunge un PDF all'indice pgvector in modo incrementale."""
    docs = _bytes_to_documents(pdf_bytes, filename)
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


def _list_bucket_pdfs() -> list[dict]:
    """Restituisce la lista dei PDF nel bucket Supabase, dal più recente al più vecchio."""
    files = get_supabase().storage.from_(BUCKET).list() or []
    return sorted(
        [f for f in files if f["name"].lower().endswith(".pdf")],
        key=lambda f: f.get("created_at") or "",
        reverse=True,
    )


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

def _pdf_row(file_info: dict, can_delete: bool = False) -> None:
    """Riga documento nel bucket: link pubblico Supabase + (opzionale) elimina."""
    name = file_info["name"]
    # get_public_url è solo costruzione di stringa, nessuna chiamata API
    url  = get_supabase().storage.from_(BUCKET).get_public_url(name)

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
            f'📄 {name}</a>'
            f'<div style="font-size:11px;color:#999;margin-top:2px;">📅 {upload_dt}</div>',
            unsafe_allow_html=True,
        )

    if can_delete and col_btn:
        with col_btn:
            if st.button("🗑️", key=f"del_{name}",
                         help=f"Rimuovi {name}", use_container_width=True):
                st.session_state.confirm_delete = name
                st.rerun()

    if st.session_state.get("confirm_delete") == name:
        st.warning(f"Eliminare **{Path(name).stem}**?")
        c_yes, c_no = st.columns(2)
        with c_yes:
            if st.button("✓ Sì", key=f"yes_{name}", use_container_width=True):
                with st.spinner("Rimozione…"):
                    get_supabase().storage.from_(BUCKET).remove([name])
                    remove_pdf_from_index(name)
                st.session_state.confirm_delete = None
                st.toast(f"«{name}» eliminato", icon="🗑️")
                st.rerun()
        with c_no:
            if st.button("✗ No", key=f"no_{name}", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()

    st.markdown(
        '<hr style="margin:4px 0;border:none;border-top:1px solid #e0e0e0;">',
        unsafe_allow_html=True,
    )


def highlight_citations(text: str) -> str:
    pattern = r"(\*?\(?\*?[\w][\w\s\.\-]+\.pdf\*?[,\s]*(?:pag\.|p\.)\s*\d+\*?\)?)"
    repl = (
        r'<span style="background:#fff3cd;padding:2px 6px;'
        r'border-radius:4px;font-size:.9em;">📄 \1</span>'
    )
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


# ── Configurazione pagina ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Oracolo delle Polizze",
    page_icon=_PILImage.open(Path(__file__).parent / "static" / "logo.png"),
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Login gate ────────────────────────────────────────────────────────────────

if not st.session_state.get("authenticated"):
    st.markdown(
        '<div style="text-align:center;padding:16px 0 8px 0">'
        '<img src="/app/static/logo.png" style="width:64px;height:64px;display:block;margin:0 auto 14px;">'
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
                    st.session_state["authenticated"] = True
                    st.session_state["username"]      = _row["username"]
                    st.session_state["name"]          = _row["name"]
                    st.session_state["role"]          = _row.get("role", "impiegato")
                    st.session_state["_open_sidebar"]  = True
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
    "last_upload_key":  None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;padding:6px 0 2px 0">'
        '<img src="/app/static/logo.png" style="width:36px;height:36px;">'
        '<span style="font-size:1.4rem;font-weight:700;color:#8B2061">Oracolo</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Gestione Documenti ────────────────────────────────────────────────────
    st.subheader("📂 Gestione Documenti")

    # — Upload (admin e impiegato) —
    if _role in ("admin", "impiegato"):
        uploaded = st.file_uploader(
            "Carica un nuovo PDF",
            type="pdf",
            help="Il documento viene caricato su Supabase e aggiunto all'indice.",
        )

        if uploaded is not None:
            _upl_key = f"{uploaded.name}_{uploaded.size}"
            if _upl_key != st.session_state.last_upload_key:
                _existing = {f["name"] for f in _list_bucket_pdfs()}
                if uploaded.name in _existing:
                    st.warning(f"**{uploaded.name}** è già presente.")
                else:
                    _pdf_bytes = uploaded.getvalue()
                    with st.spinner(f"Upload e indicizzazione di «{uploaded.name}»…"):
                        get_supabase().storage.from_(BUCKET).upload(
                            path=uploaded.name,
                            file=_pdf_bytes,
                            file_options={"content-type": "application/pdf"},
                        )
                        n = add_pdf_to_index(_pdf_bytes, uploaded.name)
                    st.session_state.last_upload_key = _upl_key
                    st.toast(
                        f"«{uploaded.name}» aggiunto — {n} pagine indicizzate",
                        icon="✅",
                    )
                    st.rerun()
                st.session_state.last_upload_key = _upl_key

        st.divider()

    # — Lista documenti —
    _pdf_files = _list_bucket_pdfs()
    if _pdf_files:
        for _fi in _pdf_files:
            _pdf_row(_fi, can_delete=(_role == "admin"))
    else:
        st.caption("Nessun documento ancora. Carica un PDF qui sopra.")

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
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:2px">'
        '<img src="/app/static/logo.png" style="width:44px;height:44px;flex-shrink:0;">'
        '<span style="font-size:2rem;font-weight:700;color:#8B2061;line-height:1.2">Oracolo delle Polizze</span>'
        '</div>'
        '<div style="color:#666;margin-top:8px;font-size:15px;line-height:1.5">'
        'Consulta i Contratti Generali di Assicurazione Sara.<br>'
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

# ── Risposta corrente ─────────────────────────────────────────────────────────

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

# ── Cronologia ────────────────────────────────────────────────────────────────

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
