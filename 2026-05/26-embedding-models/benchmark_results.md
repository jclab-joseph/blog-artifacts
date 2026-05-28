# Embedding Benchmark

## STS

| Model | Dataset | Samples | Spearman | Elapsed(s) | Total tokens | Tokens/s | Avg sec/pair |
|---|---|---:|---:|---:|---:|---:|---:|
| jhgan/ko-sbert-sts | English(STS-B val) | 1500 | 0.7762 | 31.88 | 84667 | 2655.79 | 0.02125 |
| jhgan/ko-sbert-sts | Korean(KLUE-STS val) | 519 | 0.7863 | 8.26 | 20184 | 2443.12 | 0.01592 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | English(STS-B val) | 1500 | 0.6292 | 35.90 | 103223 | 2875.46 | 0.02393 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | Korean(KLUE-STS val) | 519 | 0.7341 | 7.06 | 16447 | 2329.41 | 0.01360 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | English(STS-B val) | 1500 | 0.8747 | 7.28 | 53774 | 7387.00 | 0.00485 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Korean(KLUE-STS val) | 519 | 0.6590 | 2.87 | 20702 | 7219.38 | 0.00553 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | English(STS-B val) | 1500 | 0.8747 | 18.16 | 53774 | 2960.95 | 0.01211 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | Korean(KLUE-STS val) | 519 | 0.6589 | 9.23 | 20702 | 2241.85 | 0.01779 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | English(STS-B val) | 1500 | 0.8731 | 15.84 | 53774 | 3395.30 | 0.01056 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | Korean(KLUE-STS val) | 519 | 0.6486 | 7.37 | 20702 | 2807.42 | 0.01421 |
| Xenova/all-MiniLM-L12-v2 (original) | English(STS-B val) | 1500 | 0.8750 | 14.68 | 48709 | 3317.26 | 0.00979 |
| Xenova/all-MiniLM-L12-v2 (original) | Korean(KLUE-STS val) | 519 | 0.3142 | 17.31 | 49995 | 2888.67 | 0.03335 |
| Xenova/all-MiniLM-L12-v2 (int8) | English(STS-B val) | 1500 | 0.8711 | 15.28 | 48709 | 3188.24 | 0.01019 |
| Xenova/all-MiniLM-L12-v2 (int8) | Korean(KLUE-STS val) | 519 | 0.3012 | 19.48 | 49995 | 2566.08 | 0.03754 |
| unsloth/embeddinggemma-300m-GGUF (Q8_0) | English(STS-B val) | 1500 | 0.8800 | 1.87 | 49015 | 26198.40 | 0.00125 |
| unsloth/embeddinggemma-300m-GGUF (Q8_0) | Korean(KLUE-STS val) | 519 | 0.8630 | 0.74 | 22363 | 30413.22 | 0.00142 |
| unsloth/embeddinggemma-300m-GGUF (Q4_0) | English(STS-B val) | 1500 | 0.8812 | 1.80 | 49015 | 27236.45 | 0.00120 |
| unsloth/embeddinggemma-300m-GGUF (Q4_0) | Korean(KLUE-STS val) | 519 | 0.8592 | 0.72 | 22363 | 30941.49 | 0.00139 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 (original) | English(STS-B val) | 1500 | 0.8193 | 10.93 | 52212 | 4777.15 | 0.00729 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 (original) | Korean(KLUE-STS val) | 519 | 0.7856 | 4.66 | 25317 | 5438.42 | 0.00897 |
| Xenova/distiluse-base-multilingual-cased-v2 (int8) | English(STS-B val) | 1500 | 0.7813 | 17.63 | 52212 | 2962.25 | 0.01175 |
| Xenova/distiluse-base-multilingual-cased-v2 (int8) | Korean(KLUE-STS val) | 519 | 0.7563 | 9.69 | 25317 | 2613.16 | 0.01867 |
| jc-lab/multilingual-e5-small-ko-v2-gguf (Q8_0) | English(STS-B val) | 1500 | 0.8715 | 0.97 | 53842 | 55644.30 | 0.00065 |
| jc-lab/multilingual-e5-small-ko-v2-gguf (Q8_0) | Korean(KLUE-STS val) | 519 | 0.8040 | 0.32 | 20780 | 64254.70 | 0.00062 |
| jc-lab/multilingual-e5-small-ko-v2-gguf (Q4_K_M) | English(STS-B val) | 1500 | 0.8705 | 0.88 | 53842 | 60864.01 | 0.00059 |
| jc-lab/multilingual-e5-small-ko-v2-gguf (Q4_K_M) | Korean(KLUE-STS val) | 519 | 0.8031 | 0.33 | 20780 | 62507.24 | 0.00064 |

## Retrieval

| Model | Dataset | Queries | Docs | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Elapsed(s) | Total tokens | Tokens/s | Avg sec/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jhgan/ko-sbert-sts | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.4884 | 0.6977 | 0.7907 | 0.5973 | 231.28 | -1 | 0.00 | 5.37859 |
| jhgan/ko-sbert-sts | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1991 | 0.3649 | 0.4692 | 0.2968 | 1337.67 | -1 | 0.00 | 6.33966 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.3023 | 0.3488 | 0.4186 | 0.3407 | 235.91 | -1 | 0.00 | 5.48622 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1564 | 0.3223 | 0.4076 | 0.2469 | 1312.00 | -1 | 0.00 | 6.21799 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9535 | 0.9535 | 0.8915 | 81.84 | -1 | 0.00 | 1.90324 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.2654 | 0.4123 | 0.5071 | 0.3520 | 559.02 | -1 | 0.00 | 2.64940 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9302 | 0.9535 | 0.8857 | 184.46 | -1 | 0.00 | 4.28982 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.2322 | 0.3886 | 0.4360 | 0.3095 | 1658.68 | -1 | 0.00 | 7.86106 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.7442 | 0.9302 | 0.9535 | 0.8380 | 190.47 | -1 | 0.00 | 4.42946 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1991 | 0.3270 | 0.4028 | 0.2716 | 1727.12 | -1 | 0.00 | 8.18540 |
| Xenova/all-MiniLM-L12-v2 (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9302 | 1.0000 | 1.0000 | 0.9651 | 154.73 | -1 | 0.00 | 3.59842 |
| Xenova/all-MiniLM-L12-v2 (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.0284 | 0.0379 | 0.0521 | 0.0355 | 1580.13 | -1 | 0.00 | 7.48878 |
| Xenova/all-MiniLM-L12-v2 (int8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8837 | 0.9767 | 1.0000 | 0.9271 | 168.95 | -1 | 0.00 | 3.92914 |
| Xenova/all-MiniLM-L12-v2 (int8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.0237 | 0.0379 | 0.0427 | 0.0302 | 1729.01 | -1 | 0.00 | 8.19435 |
| unsloth/embeddinggemma-300m-GGUF (Q8_0) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8140 | 0.9767 | 0.9767 | 0.8837 | 20.33 | -1 | 0.00 | 0.47275 |
| unsloth/embeddinggemma-300m-GGUF (Q8_0) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.6066 | 0.8057 | 0.8768 | 0.7101 | 211.36 | -1 | 0.00 | 1.00168 |
| unsloth/embeddinggemma-300m-GGUF (Q4_0) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8605 | 0.9767 | 0.9767 | 0.9147 | 20.09 | -1 | 0.00 | 0.46724 |
| unsloth/embeddinggemma-300m-GGUF (Q4_0) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5592 | 0.7962 | 0.8531 | 0.6787 | 209.15 | -1 | 0.00 | 0.99121 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.5116 | 0.6512 | 0.7674 | 0.5953 | 98.78 | -1 | 0.00 | 2.29715 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1469 | 0.2133 | 0.2844 | 0.1906 | 665.74 | -1 | 0.00 | 3.15514 |
| Xenova/distiluse-base-multilingual-cased-v2 (int8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.5349 | 0.7674 | 0.7674 | 0.6279 | 214.55 | -1 | 0.00 | 4.98956 |
| Xenova/distiluse-base-multilingual-cased-v2 (int8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1564 | 0.2275 | 0.3081 | 0.2055 | 1886.81 | -1 | 0.00 | 8.94223 |
| jc-lab/multilingual-e5-small-ko-v2-gguf (Q8_0) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8140 | 0.9767 | 1.0000 | 0.8961 | 9.35 | -1 | 0.00 | 0.21737 |
| jc-lab/multilingual-e5-small-ko-v2-gguf (Q8_0) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5592 | 0.8009 | 0.8626 | 0.6812 | 90.83 | -1 | 0.00 | 0.43047 |
| jc-lab/multilingual-e5-small-ko-v2-gguf (Q4_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9767 | 1.0000 | 0.9078 | 9.57 | -1 | 0.00 | 0.22258 |
| jc-lab/multilingual-e5-small-ko-v2-gguf (Q4_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5545 | 0.7867 | 0.8626 | 0.6738 | 91.60 | -1 | 0.00 | 0.43413 |
