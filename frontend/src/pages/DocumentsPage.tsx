/**
 * DocumentsPage — corpus management.
 *
 * Fetches the indexed library from GET /api/documents on mount and after
 * every successful upload. Uploads stream through DocumentUpload, which
 * calls back into us so we can refetch.
 */

import { useCallback, useEffect, useState } from "react";

import { DocumentUpload } from "../components/DocumentUpload";

type IndexedDoc = {
  source: string;
  chunks: number;
  pages: number;
};

type Library = {
  documents: IndexedDoc[];
  total_chunks: number;
};

export function DocumentsPage() {
  const [library, setLibrary] = useState<Library | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) throw new Error(`Failed: ${res.status}`);
      setLibrary(await res.json());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load library");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const totalPages = library?.documents.reduce((a, d) => a + d.pages, 0) ?? 0;

  return (
    <div className="mx-auto flex h-full max-w-4xl flex-col gap-8 overflow-y-auto px-10 py-12">
      <header>
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-400">
          Workspace
        </div>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">
          Documents
        </h1>
        <p className="mt-2 max-w-xl text-[13px] leading-relaxed text-slate-500">
          Upload PDFs to build your analysis corpus. The agent retrieves
          passages from these documents to answer your questions.
        </p>
      </header>

      {/* Stats strip */}
      <div className="grid grid-cols-3 gap-3">
        <Stat label="Documents" value={library?.documents.length ?? 0} />
        <Stat label="Pages indexed" value={totalPages} />
        <Stat label="Chunks in vector DB" value={library?.total_chunks ?? 0} />
      </div>

      {/* Upload card */}
      <section className="rounded-xl border border-slate-200/70 bg-white p-5 shadow-sm">
        <div className="mb-4">
          <h2 className="text-[13px] font-semibold tracking-tight text-slate-900">
            Upload a new document
          </h2>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Parsed with pdfplumber, chunked at ~700 tokens, embedded with
            text-embedding-3-small, stored in Qdrant.
          </p>
        </div>
        <DocumentUpload onUploadComplete={refresh} />
      </section>

      {/* Library */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-[13px] font-semibold tracking-tight text-slate-900">
            Library
          </h2>
          <button
            onClick={() => void refresh()}
            className="flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:border-slate-300 hover:bg-slate-50"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-3 w-3"
            >
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            Refresh
          </button>
        </div>

        {loading ? (
          <LibrarySkeleton />
        ) : error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-[12px] text-red-700">
            {error}
          </div>
        ) : library?.documents.length === 0 ? (
          <EmptyLibrary />
        ) : (
          <ul className="divide-y divide-slate-200/70 overflow-hidden rounded-xl border border-slate-200/70 bg-white shadow-sm">
            {library?.documents.map((doc) => (
              <DocRow key={doc.source} doc={doc} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-slate-200/70 bg-white p-4 shadow-sm">
      <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-400">
        {label}
      </div>
      <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums tracking-tight text-slate-900">
        {value.toLocaleString()}
      </div>
    </div>
  );
}

function DocRow({ doc }: { doc: IndexedDoc }) {
  return (
    <li className="group flex items-center gap-3 px-4 py-3 transition-colors hover:bg-slate-50/60">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-slate-200 bg-white">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-4 w-4 text-slate-500"
        >
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      </div>
      <div className="min-w-0 flex-1">
        <div
          className="truncate text-[12.5px] font-medium text-slate-900"
          title={doc.source}
        >
          {doc.source}
        </div>
        <div className="mt-0.5 flex items-center gap-2 font-mono text-[10px] tabular-nums text-slate-500">
          <span>{doc.pages} pages</span>
          <span className="h-0.5 w-0.5 rounded-full bg-slate-300" />
          <span>{doc.chunks} chunks</span>
        </div>
      </div>
      <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700 ring-1 ring-inset ring-emerald-200/60">
        indexed
      </span>
    </li>
  );
}

function LibrarySkeleton() {
  return (
    <ul className="divide-y divide-slate-200/70 overflow-hidden rounded-xl border border-slate-200/70 bg-white">
      {[0, 1, 2].map((i) => (
        <li key={i} className="flex items-center gap-3 px-4 py-3">
          <div className="h-8 w-8 animate-pulse rounded-md bg-slate-100" />
          <div className="flex-1 space-y-1.5">
            <div className="h-3 w-2/3 animate-pulse rounded bg-slate-100" />
            <div className="h-2.5 w-1/3 animate-pulse rounded bg-slate-100" />
          </div>
        </li>
      ))}
    </ul>
  );
}

function EmptyLibrary() {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-12 text-center">
      <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-slate-100">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-5 w-5 text-slate-400"
        >
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      </div>
      <h3 className="text-[13px] font-semibold text-slate-900">
        No documents yet
      </h3>
      <p className="mt-1 text-[11px] text-slate-500">
        Upload a PDF above to start building your corpus.
      </p>
    </div>
  );
}
