import os
import sys
from pathlib import Path

import pymupdf as fitz
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
DOCUMENTS_DIR = BASE_DIR / "documents"
STORAGE_DIR = BASE_DIR / "storage"

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
        model="claude-sonnet-4-6",
        system_prompt=SYSTEM_PROMPT,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.node_parser = SPLITTER


def load_pdfs_with_pymupdf(docs_dir: Path) -> list[Document]:
    documents = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            if text.strip():
                documents.append(Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "page_label": str(page_num + 1),
                    },
                ))
        num_pages = len(doc)
        doc.close()
        print(f"  Caricato: {pdf_path.name} ({num_pages} pagine)")
    return documents


def build_and_save_index() -> VectorStoreIndex:
    print("Indicizzazione documenti in corso...")
    documents = load_pdfs_with_pymupdf(DOCUMENTS_DIR)
    if not documents:
        raise ValueError("Nessun testo estratto dai PDF. Verifica che i file non siano solo immagini.")
    print(f"  Pagine con testo: {len(documents)}")
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[SPLITTER],
        show_progress=True,
    )
    STORAGE_DIR.mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print(f"Indice salvato in {STORAGE_DIR}\n")
    return index


def load_existing_index() -> VectorStoreIndex:
    print("Caricamento indice esistente da storage/...")
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_context)


def get_index() -> VectorStoreIndex:
    storage_has_data = STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir())
    if storage_has_data:
        return load_existing_index()

    if not list(DOCUMENTS_DIR.glob("*.pdf")):
        raise FileNotFoundError(
            f"Nessun PDF trovato in '{DOCUMENTS_DIR}'. "
            "Aggiungi i file CGA e riavvia."
        )

    return build_and_save_index()


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
