# Embedding STS Benchmark

| Model | Dataset | Samples | Spearman | Elapsed(s) | Total tokens | Tokens/s | Avg sec/pair |
|---|---|---:|---:|---:|---:|---:|---:|
| jhgan/ko-sbert-sts | English(STS-B val) | 1500 | 0.7762 | 41.30 | 84667 | 2050.22 | 0.02753 |
| jhgan/ko-sbert-sts | Korean(KLUE-STS val) | 519 | 0.7863 | 10.89 | 20184 | 1852.63 | 0.02099 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | English(STS-B val) | 1500 | 0.6292 | 48.66 | 103223 | 2121.15 | 0.03244 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | Korean(KLUE-STS val) | 519 | 0.7341 | 9.37 | 16447 | 1755.90 | 0.01805 |
| paraphrase-multilingual-MiniLM-L12-v2 | English(STS-B val) | 1500 | 0.8747 | 8.35 | 53774 | 6437.41 | 0.00557 |
| paraphrase-multilingual-MiniLM-L12-v2 | Korean(KLUE-STS val) | 519 | 0.6590 | 3.55 | 20702 | 5832.06 | 0.00684 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | English(STS-B val) | 1500 | 0.8747 | 15.47 | 53774 | 3475.86 | 0.01031 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | Korean(KLUE-STS val) | 519 | 0.6589 | 9.52 | 20702 | 2175.41 | 0.01834 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | English(STS-B val) | 1500 | 0.8731 | 15.38 | 53774 | 3496.04 | 0.01025 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | Korean(KLUE-STS val) | 519 | 0.6505 | 9.30 | 20702 | 2226.92 | 0.01791 |
| Xenova/all-MiniLM-L12-v2 (original) | English(STS-B val) | 1500 | 0.8750 | 12.80 | 48709 | 3805.89 | 0.00853 |
| Xenova/all-MiniLM-L12-v2 (original) | Korean(KLUE-STS val) | 519 | 0.3142 | 33.11 | 49995 | 1510.08 | 0.06379 |
| Xenova/all-MiniLM-L12-v2 (fp16) | English(STS-B val) | 1500 | 0.8750 | 14.63 | 48709 | 3328.82 | 0.00975 |
| Xenova/all-MiniLM-L12-v2 (fp16) | Korean(KLUE-STS val) | 519 | 0.3141 | 39.41 | 49995 | 1268.62 | 0.07593 |
| Xenova/all-MiniLM-L12-v2 (int8) | English(STS-B val) | 1500 | 0.8706 | 13.55 | 48709 | 3595.29 | 0.00903 |
| Xenova/all-MiniLM-L12-v2 (int8) | Korean(KLUE-STS val) | 519 | 0.2973 | 33.15 | 49995 | 1507.96 | 0.06388 |
