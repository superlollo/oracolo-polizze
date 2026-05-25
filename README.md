# Oracolo Polizze

Sistema RAG per l'analisi e la consultazione delle Condizioni Generali di Assicurazione (CGA) tramite AI.

## Struttura

```
oracolo-polizze/
├── documents/    # PDF dei CGA da indicizzare
├── src/          # Codice sorgente
├── .env          # API key (non condividere mai)
├── requirements.txt
└── README.md
```

## Setup

1. Inserisci le tue API key nel file `.env`
2. Aggiungi i PDF dei CGA nella cartella `documents/`
3. Installa le dipendenze:

```bash
pip install -r requirements.txt
```
