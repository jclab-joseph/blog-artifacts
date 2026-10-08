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
| second-state/embeddinggemma-300m-GGUF (Q4_K_M) | English(STS-B val) | 1500 | 0.8786 | 11.25 | 49015 | 4355.46 | 0.00750 |
| second-state/embeddinggemma-300m-GGUF (Q4_K_M) | Korean(KLUE-STS val) | 519 | 0.8602 | 0.94 | 22363 | 23797.79 | 0.00181 |
| second-state/embeddinggemma-300m-GGUF (Q5_K_M) | English(STS-B val) | 1500 | 0.8800 | 10.70 | 49015 | 4579.41 | 0.00714 |
| second-state/embeddinggemma-300m-GGUF (Q5_K_M) | Korean(KLUE-STS val) | 519 | 0.8552 | 0.93 | 22363 | 24154.46 | 0.00178 |
| mykor/harrier-oss-v1-270m-GGUF (Q4_K_M) | English(STS-B val) | 1500 | 0.7538 | 18.89 | 49015 | 2594.33 | 0.01260 |
| mykor/harrier-oss-v1-270m-GGUF (Q4_K_M) | Korean(KLUE-STS val) | 519 | 0.7290 | 4.43 | 22363 | 5042.78 | 0.00854 |
| mykor/harrier-oss-v1-270m-GGUF (Q5_K_M) | English(STS-B val) | 1500 | 0.7537 | 18.48 | 49015 | 2652.94 | 0.01232 |
| mykor/harrier-oss-v1-270m-GGUF (Q5_K_M) | Korean(KLUE-STS val) | 519 | 0.7305 | 4.43 | 22363 | 5051.53 | 0.00853 |
| mykor/harrier-oss-v1-270m-GGUF (Q8) | English(STS-B val) | 1500 | 0.7558 | 18.61 | 49015 | 2634.50 | 0.01240 |
| mykor/harrier-oss-v1-270m-GGUF (Q8) | Korean(KLUE-STS val) | 519 | 0.7320 | 4.41 | 22363 | 5070.89 | 0.00850 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q5_K_M) | English(STS-B val) | 1500 | 0.8372 | 10.22 | 46015 | 4503.67 | 0.00681 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q5_K_M) | Korean(KLUE-STS val) | 519 | 0.8240 | 0.98 | 21325 | 21681.62 | 0.00190 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q4_K_M) | English(STS-B val) | 1500 | 0.8348 | 10.20 | 46015 | 4510.78 | 0.00680 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q4_K_M) | Korean(KLUE-STS val) | 519 | 0.8230 | 0.96 | 21325 | 22139.07 | 0.00186 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q8) | English(STS-B val) | 1500 | 0.8366 | 10.23 | 46015 | 4496.90 | 0.00682 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q8) | Korean(KLUE-STS val) | 519 | 0.8241 | 0.95 | 21325 | 22383.34 | 0.00184 |
| google/embeddinggemma-300m | English(STS-B val) | 1500 | 0.8665 | 6.44 | 49015 | 7606.57 | 0.00430 |
| google/embeddinggemma-300m | Korean(KLUE-STS val) | 519 | 0.8607 | 1.56 | 22363 | 14325.63 | 0.00301 |
| dragonkue/BGE-m3-ko (original) | English(STS-B val) | 1500 | 0.8722 | 6.38 | 53774 | 8423.26 | 0.00426 |
| dragonkue/BGE-m3-ko (original) | Korean(KLUE-STS val) | 519 | 0.8867 | 2.47 | 20702 | 8366.88 | 0.00477 |
| Neuwhufbox/BGE-m3-ko-gguf (Q8_0) | English(STS-B val) | 1500 | 0.8724 | 24.50 | 53774 | 2194.88 | 0.01633 |
| Neuwhufbox/BGE-m3-ko-gguf (Q8_0) | Korean(KLUE-STS val) | 519 | 0.8864 | 2.26 | 20702 | 9160.32 | 0.00435 |
| codefuse-ai/F2LLM-v2-1.7B (original) | English(STS-B val) | 1500 | 0.8792 | 7.87 | 46267 | 5876.08 | 0.00525 |
| codefuse-ai/F2LLM-v2-1.7B (original) | Korean(KLUE-STS val) | 519 | 0.8623 | 3.50 | 25615 | 7312.00 | 0.00675 |
| mradermacher/F2LLM-v2-1.7B-GGUF (Q8_0) | English(STS-B val) | 1500 | 0.8790 | 18.60 | 43267 | 2326.40 | 0.01240 |
| mradermacher/F2LLM-v2-1.7B-GGUF (Q8_0) | Korean(KLUE-STS val) | 519 | 0.8619 | 5.05 | 24577 | 4870.91 | 0.00972 |
| mradermacher/F2LLM-v2-1.7B-GGUF (Q4_K_M) | English(STS-B val) | 1500 | 0.8327 | 19.15 | 43267 | 2259.90 | 0.01276 |
| mradermacher/F2LLM-v2-1.7B-GGUF (Q4_K_M) | Korean(KLUE-STS val) | 519 | 0.7641 | 5.48 | 24577 | 4482.41 | 0.01056 |
| mradermacher/Qwen3-Embedding-8B-i1-GGUF (i1-Q4_K_M) | English(STS-B val) | 1500 | 0.8993 | 42.92 | 43267 | 1008.07 | 0.02861 |
| mradermacher/Qwen3-Embedding-8B-i1-GGUF (i1-Q4_K_M) | Korean(KLUE-STS val) | 519 | 0.8653 | 14.06 | 24577 | 1748.24 | 0.02709 |
| upstage/solar-embedding-2 (query+passage) | English(STS-B val) | 1500 | 0.8579 | 31.69 | 43267 | 1365.51 | 0.02112 |
| upstage/solar-embedding-2 (query+passage) | Korean(KLUE-STS val) | 519 | 0.8100 | 11.53 | 24577 | 2132.27 | 0.02221 |
| google/embeddinggemma-2 (original) | English(STS-B val) | 1500 | 0.8470 | 2.28 | 49015 | 21483.97 | 0.00152 |
| google/embeddinggemma-2 (original) | Korean(KLUE-STS val) | 519 | 0.8203 | 0.73 | 22363 | 30843.13 | 0.00140 |
| unsloth/embeddinggemma-2-GGUF (Q8_0) | English(STS-B val) | 1500 | 0.8469 | 13.88 | 49015 | 3530.09 | 0.00926 |
| unsloth/embeddinggemma-2-GGUF (Q8_0) | Korean(KLUE-STS val) | 519 | 0.8198 | 0.83 | 22363 | 27013.78 | 0.00160 |
| unsloth/embeddinggemma-2-GGUF (UD-Q4_K_XL) | English(STS-B val) | 1500 | 0.8468 | 13.63 | 49015 | 3595.14 | 0.00909 |
| unsloth/embeddinggemma-2-GGUF (UD-Q4_K_XL) | Korean(KLUE-STS val) | 519 | 0.8175 | 0.83 | 22363 | 26792.17 | 0.00161 |

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
| second-state/embeddinggemma-300m-GGUF (Q4_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.7907 | 0.9535 | 0.9767 | 0.8690 | 31.32 | -1 | 0.00 | 0.72831 |
| second-state/embeddinggemma-300m-GGUF (Q4_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.6114 | 0.8246 | 0.8578 | 0.7160 | 257.48 | -1 | 0.00 | 1.22030 |
| second-state/embeddinggemma-300m-GGUF (Q5_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8605 | 0.9767 | 0.9767 | 0.9070 | 30.31 | -1 | 0.00 | 0.70489 |
| second-state/embeddinggemma-300m-GGUF (Q5_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.6161 | 0.8199 | 0.8626 | 0.7166 | 260.04 | -1 | 0.00 | 1.23240 |
| mykor/harrier-oss-v1-270m-GGUF (Q4_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.7209 | 0.8605 | 0.9070 | 0.7895 | 111.97 | -1 | 0.00 | 2.60389 |
| mykor/harrier-oss-v1-270m-GGUF (Q4_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.2607 | 0.4502 | 0.5640 | 0.3705 | 1203.84 | -1 | 0.00 | 5.70543 |
| mykor/harrier-oss-v1-270m-GGUF (Q5_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.6977 | 0.8372 | 0.9535 | 0.7891 | 112.05 | -1 | 0.00 | 2.60572 |
| mykor/harrier-oss-v1-270m-GGUF (Q5_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.2749 | 0.4550 | 0.5592 | 0.3783 | 1205.43 | -1 | 0.00 | 5.71293 |
| mykor/harrier-oss-v1-270m-GGUF (Q8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.7209 | 0.8605 | 0.9302 | 0.7981 | 111.39 | -1 | 0.00 | 2.59041 |
| mykor/harrier-oss-v1-270m-GGUF (Q8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.2701 | 0.4739 | 0.5687 | 0.3789 | 1200.63 | -1 | 0.00 | 5.69021 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q5_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.7674 | 0.9070 | 0.9535 | 0.8426 | 33.88 | -1 | 0.00 | 0.78781 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q5_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.4360 | 0.6114 | 0.6825 | 0.5314 | 316.87 | -1 | 0.00 | 1.50177 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q4_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.7674 | 0.9070 | 0.9535 | 0.8438 | 32.47 | -1 | 0.00 | 0.75517 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q4_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.4597 | 0.6161 | 0.6825 | 0.5423 | 315.73 | -1 | 0.00 | 1.49636 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.7674 | 0.9302 | 0.9302 | 0.8411 | 32.42 | -1 | 0.00 | 0.75397 |
| mykor/granite-embedding-311m-multilingual-r2-GGUF (Q8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.4455 | 0.6114 | 0.6872 | 0.5365 | 309.73 | -1 | 0.00 | 1.46792 |
| google/embeddinggemma-300m | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9070 | 0.9535 | 0.8775 | 19.17 | -1 | 0.00 | 0.44574 |
| google/embeddinggemma-300m | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5545 | 0.8009 | 0.8626 | 0.6764 | 221.04 | -1 | 0.00 | 1.04757 |
| dragonkue/BGE-m3-ko (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8837 | 1.0000 | 1.0000 | 0.9380 | 66.52 | -1 | 0.00 | 1.54703 |
| dragonkue/BGE-m3-ko (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.6540 | 0.8389 | 0.9005 | 0.7546 | 588.69 | -1 | 0.00 | 2.79001 |
| Neuwhufbox/BGE-m3-ko-gguf (Q8_0) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8605 | 1.0000 | 1.0000 | 0.9264 | 69.66 | -1 | 0.00 | 1.62005 |
| Neuwhufbox/BGE-m3-ko-gguf (Q8_0) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.6445 | 0.8436 | 0.9005 | 0.7502 | 535.17 | -1 | 0.00 | 2.53637 |
| codefuse-ai/F2LLM-v2-1.7B (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9302 | 1.0000 | 1.0000 | 0.9612 | 72.17 | -1 | 0.00 | 1.67842 |
| codefuse-ai/F2LLM-v2-1.7B (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5261 | 0.7678 | 0.8246 | 0.6523 | 673.87 | -1 | 0.00 | 3.19369 |
| mradermacher/F2LLM-v2-1.7B-GGUF (Q8_0) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9302 | 1.0000 | 1.0000 | 0.9612 | 111.19 | -1 | 0.00 | 2.58579 |
| mradermacher/F2LLM-v2-1.7B-GGUF (Q8_0) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5213 | 0.7725 | 0.8246 | 0.6502 | 1414.90 | -1 | 0.00 | 6.70569 |
| mradermacher/F2LLM-v2-1.7B-GGUF (Q4_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9302 | 0.9767 | 0.9767 | 0.9535 | 117.73 | -1 | 0.00 | 2.73802 |
| mradermacher/F2LLM-v2-1.7B-GGUF (Q4_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.4455 | 0.6635 | 0.7299 | 0.5610 | 1452.93 | -1 | 0.00 | 6.88594 |
| mradermacher/Qwen3-Embedding-8B-i1-GGUF (i1-Q4_K_M) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9767 | 1.0000 | 1.0000 | 0.9884 | 322.83 | -1 | 0.00 | 7.50762 |
| mradermacher/Qwen3-Embedding-8B-i1-GGUF (i1-Q4_K_M) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.6445 | 0.8389 | 0.8863 | 0.7428 | 4125.11 | -1 | 0.00 | 19.55028 |
| upstage/solar-embedding-2 (query+passage) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9070 | 0.9767 | 1.0000 | 0.9438 | 121.18 | -1 | 0.00 | 2.81807 |
| upstage/solar-embedding-2 (query+passage) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5972 | 0.7867 | 0.8673 | 0.7064 | 1549.09 | -1 | 0.00 | 7.34166 |
| google/embeddinggemma-2 (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9302 | 0.9767 | 0.8903 | 13.14 | -1 | 0.00 | 0.30564 |
| google/embeddinggemma-2 (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5308 | 0.7725 | 0.8578 | 0.6537 | 154.80 | -1 | 0.00 | 0.73364 |
| unsloth/embeddinggemma-2-GGUF (Q8_0) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9302 | 0.9767 | 0.8903 | 34.27 | -1 | 0.00 | 0.79687 |
| unsloth/embeddinggemma-2-GGUF (Q8_0) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5166 | 0.7773 | 0.8483 | 0.6453 | 216.21 | -1 | 0.00 | 1.02467 |
| unsloth/embeddinggemma-2-GGUF (UD-Q4_K_XL) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9535 | 0.9767 | 0.8973 | 34.85 | -1 | 0.00 | 0.81057 |
| unsloth/embeddinggemma-2-GGUF (UD-Q4_K_XL) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5024 | 0.7725 | 0.8389 | 0.6370 | 218.37 | -1 | 0.00 | 1.03491 |

## 참고

- second-state/embeddinggemma-300m-GGUF, harrier-oss-v1-270m-GGUF, granite-embedding-311m-multilingual-r2-GGUF 는 RTX 3080 에서 돌렸습니다.
- google/embeddinggemma-300m 은 RTX 3090 에서 돌렸습니다.
- mradermacher/Qwen3-Embedding-8B-i1-GGUF, dragonkue/BGE-m3-ko, Neuwhufbox/BGE-m3-ko-gguf, codefuse-ai/F2LLM-v2-1.7B, mradermacher/F2LLM-v2-1.7B-GGUF 는 RTX 3080 에서 돌렸습니다.
- 신규 모델의 입력은 256 토큰으로 잘랐고, Qwen3-Embedding 은 query 에 `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`, F2LLM 은 `Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: ` 를 붙였습니다 (retrieval 만 해당). pooling 은 Qwen3/F2LLM=last token, BGE-m3-ko=CLS.
- OpenRouter(Qwen3-Embedding-8B/4B) 는 소요 시간 문제로 제외했습니다.
- upstage/solar-embedding-2 는 Upstage API 로 돌렸습니다. retrieval 은 query 에 `solar-embedding-2-query`, 문서에 `solar-embedding-2-passage` 를 썼고, STS 는 양쪽 문장 모두 `solar-embedding-2-passage` 를 썼습니다. query 지시문은 서버 측에서 붙으므로 prefix 는 붙이지 않았습니다. 입력은 Qwen3 토크나이저(API 토큰 수와 일치)로 256 토큰으로 잘랐고, 소요 시간은 API rate limit(100 RPM / 300K TPM)에 맞춘 페이싱이 포함된 값입니다.
- google/embeddinggemma-2, unsloth/embeddinggemma-2-GGUF 는 RTX 3090 에서 돌렸습니다. 입력은 256 토큰으로 잘랐고, 모델 카드의 `task: search result | query: ` (query) / `title: none | text: ` (문서) prefix 를 썼습니다 (retrieval 만 해당). GGUF 는 llama-cpp-python 0.3.36 에 llama.cpp `4fbc76dec5` (gemma-embedding2 지원 커밋)를 넣어 빌드했고, 출력 차원(768)이 hidden(512)과 달라 `llama_model_n_embd_out` 으로 크기를 맞췄습니다.
