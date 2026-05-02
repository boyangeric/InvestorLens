/**
 * DocumentUpload — drag/drop or click-to-pick PDF uploader.
 *
 * Hits POST /api/documents/upload (proxied to backend in dev). The backend
 * runs parse → chunk → embed → store inline. On success, calls back to the
 * parent (DocumentsPage) so the library can refresh from the server.
 */

import { useRef, useState } from "react";

type UploadResult = {
  filename: string;
  pages: number;
  chunks: number;
};

type Props = {
  onUploadComplete?: (result: UploadResult) => void;
};

export function DocumentUpload({ onUploadComplete }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<UploadResult | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }

    setError(null);
    setLastResult(null);
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Upload failed (${res.status})`);
      }

      const result = (await res.json()) as UploadResult;
      setLastResult(result);
      onUploadComplete?.(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-3">
      <div
        onClick={() => !uploading && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!uploading) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (uploading) return;
          const file = e.dataTransfer.files[0];
          if (file) void handleFile(file);
        }}
        className={`relative cursor-pointer overflow-hidden rounded-xl border border-dashed px-6 py-10 text-center transition-all ${
          dragOver
            ? "border-slate-400 bg-slate-50"
            : uploading
              ? "cursor-wait border-slate-200 bg-slate-50/60"
              : "border-slate-300 hover:border-slate-400 hover:bg-slate-50/60"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleFile(file);
            e.target.value = "";
          }}
        />

        {uploading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="h-7 w-7 animate-spin rounded-full border-2 border-slate-200 border-t-slate-700" />
            <p className="text-[12px] font-medium text-slate-700">
              Parsing, chunking, embedding…
            </p>
          </div>
        ) : (
          <>
            <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg border border-slate-200 bg-white shadow-sm">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-5 w-5 text-slate-500"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="text-[13px] font-medium text-slate-800">
              Drop a PDF here or{" "}
              <span className="text-slate-900 underline decoration-slate-300 underline-offset-2">
                click to browse
              </span>
            </p>
            <p className="mt-1 text-[11px] text-slate-500">
              Annual reports, 10-Ks, ETF factsheets, fund PDS
            </p>
          </>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {error}
        </div>
      )}

      {lastResult && !uploading && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-800">
          <span>✓</span>
          <span>
            <span className="font-medium">{lastResult.filename}</span> ingested —{" "}
            {lastResult.pages} pages, {lastResult.chunks} chunks
          </span>
        </div>
      )}
    </div>
  );
}
