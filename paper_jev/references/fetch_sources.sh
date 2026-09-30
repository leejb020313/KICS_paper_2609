#!/usr/bin/env bash
# Download the cited source PDFs into pdfs/ (not committed: third-party copyright). verify_references.py needs them.
cd "$(dirname "$0")" && mkdir -p pdfs && cd pdfs
for id in 2608.30731v2 2609.26550v3 2609.29769v2 2609.11446v1; do
  curl -sfL -o "${id%v*}.pdf" "https://arxiv.org/pdf/$id"
done
curl -sfL -o clef2024_task1_overview.pdf https://ceur-ws.org/Vol-3740/paper-24.pdf
curl -sfL -o icwsm2020_claimbuster.pdf https://ojs.aaai.org/index.php/ICWSM/article/download/7346/7200
ls -1
