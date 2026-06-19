import os
import sys
import tempfile
from pathlib import Path

import pymupdf as fitz
import sqlalchemy as sa
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.supabase import SupabaseVectorStore
from supabase import create_client

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

BUCKET = "cga-documents"

SYSTEM_PROMPT = (
    "Sei l'Oracolo delle Polizze dell'Agenzia Sara Assicurazioni."
    "Stai parlando con un agente assicurativo o subagente assicurativo o impiegato, NON con un cliente finale."
    "Le tue risposte devono essere tecniche, precise e operative."
    "Cita sempre il nome del documento e la pagina."
    "Usa linguaggio professionale da operatore del settore, non da sportello clienti."
    "Se la risposta ha implicazioni per il cliente finale, segnalalo come nota separata."
    "Se non trovi l'informazione, dì esplicitamente che non è presente nei documenti. "
    "Non inventare mai coperture o esclusioni."
)

SPLITTER = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def configure_settings():
    Settings.llm = Anthropic(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system_prompt=SYSTEM_PROMPT,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.node_parser = SPLITTER


def load_pdfs_from_supabase_storage() -> list[Document]:
    """Scarica i PDF dal bucket Supabase ed estrae il testo pagina per pagina."""
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    files = sb.storage.from_(BUCKET).list()
    pdfs = [f for f in files if f["name"].lower().endswith(".pdf")]

    if not pdfs:
        return []

    documents: list[Document] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for file_info in pdfs:
            name = file_info["name"]
            print(f"  Download: {name} …")
            data = sb.storage.from_(BUCKET).download(name)

            pdf_path = tmp / name
            pdf_path.write_bytes(data)

            fitz_doc = fitz.open(str(pdf_path))
            num_pages = len(fitz_doc)
            for page_num in range(num_pages):
                text = fitz_doc[page_num].get_text()
                if text.strip():
                    documents.append(Document(
                        text=text,
                        metadata={
                            "file_name":  name,
                            "page_label": str(page_num + 1),
                        },
                    ))
            fitz_doc.close()
            print(f"  Caricato: {name} ({num_pages} pagine)")

    return documents


def index_is_empty() -> bool:
    """Restituisce True se vecs.documents non ha ancora record."""
    engine = sa.create_engine(os.environ["SUPABASE_DB_URL"])
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT 1 FROM vecs.documents LIMIT 1")
            ).first()
            return row is None
    except Exception:
        return True          # tabella assente → da costruire
    finally:
        engine.dispose()


def _make_vector_store() -> SupabaseVectorStore:
    return SupabaseVectorStore(
        postgres_connection_string=os.environ["SUPABASE_DB_URL"],
        collection_name="documents",
        dimension=1536,      # text-embedding-3-small
    )


def build_and_save_index() -> VectorStoreIndex:
    print("Indicizzazione documenti in corso...")
    documents = load_pdfs_from_supabase_storage()
    if not documents:
        raise ValueError("Nessun testo estratto. Verifica che il bucket non sia vuoto.")
    print(f"  Pagine con testo: {len(documents)}")

    storage_context = StorageContext.from_defaults(
        vector_store=_make_vector_store()
    )
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[SPLITTER],
        show_progress=True,
    )
    print("Indice salvato su Supabase (pgvector)\n")
    return index


def load_existing_index() -> VectorStoreIndex:
    print("Caricamento indice esistente da Supabase (pgvector)...")
    return VectorStoreIndex.from_vector_store(_make_vector_store())


def get_index() -> VectorStoreIndex:
    if index_is_empty():
        return build_and_save_index()
    return load_existing_index()


def main():
    configure_settings()

    try:
        index = get_index()
    except (FileNotFoundError, ValueError) as e:
        print(f"Errore: {e}")
        return

    query_engine = index.as_query_engine(similarity_top_k=5)

    print("Oracolo delle Polizze pronto.")
    print("Scrivi la tua domanda o 'esci' per uscire.\n")

    while True:
        try:
            domanda = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nArrivederci.")
            break

        if domanda.lower() == "esci":
            print("Arrivederci.")
            break

        if not domanda:
            continue

        risposta = query_engine.query(domanda)
        print(f"\nOracolo: {risposta}\n")


if __name__ == "__main__":
    main()
