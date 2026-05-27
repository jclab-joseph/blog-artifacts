# Embedding Benchmark

## STS

| Model | Dataset | Samples | Spearman | Elapsed(s) | Total tokens | Tokens/s | Avg sec/pair |
|---|---|---:|---:|---:|---:|---:|---:|
| jhgan/ko-sbert-sts | English(STS-B val) | 1500 | 0.7762 | 22.43 | 84667 | 3775.06 | 0.01495 |
| jhgan/ko-sbert-sts | Korean(KLUE-STS val) | 519 | 0.7863 | 6.11 | 20184 | 3302.93 | 0.01177 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | English(STS-B val) | 1500 | 0.6292 | 21.76 | 103223 | 4743.25 | 0.01451 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | Korean(KLUE-STS val) | 519 | 0.7341 | 4.26 | 16447 | 3858.76 | 0.00821 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | English(STS-B val) | 1500 | 0.8747 | 5.29 | 53774 | 10172.17 | 0.00352 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Korean(KLUE-STS val) | 519 | 0.6590 | 2.06 | 20702 | 10044.36 | 0.00397 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | English(STS-B val) | 1500 | 0.8747 | 13.50 | 53774 | 3983.99 | 0.00900 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | Korean(KLUE-STS val) | 519 | 0.6589 | 6.30 | 20702 | 3286.21 | 0.01214 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | English(STS-B val) | 1500 | 0.8735 | 14.43 | 53774 | 3727.37 | 0.00962 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | Korean(KLUE-STS val) | 519 | 0.6505 | 7.57 | 20702 | 2736.31 | 0.01458 |
| Xenova/all-MiniLM-L12-v2 (original) | English(STS-B val) | 1500 | 0.8750 | 10.66 | 48709 | 4570.92 | 0.00710 |
| Xenova/all-MiniLM-L12-v2 (original) | Korean(KLUE-STS val) | 519 | 0.3142 | 13.65 | 49995 | 3662.73 | 0.02630 |
| Xenova/all-MiniLM-L12-v2 (fp16) | English(STS-B val) | 1500 | 0.8750 | 11.39 | 48709 | 4278.17 | 0.00759 |
| Xenova/all-MiniLM-L12-v2 (fp16) | Korean(KLUE-STS val) | 519 | 0.3140 | 16.32 | 49995 | 3062.63 | 0.03145 |
| Xenova/all-MiniLM-L12-v2 (int8) | English(STS-B val) | 1500 | 0.8708 | 13.63 | 48709 | 3573.86 | 0.00909 |
| Xenova/all-MiniLM-L12-v2 (int8) | Korean(KLUE-STS val) | 519 | 0.2969 | 20.91 | 49995 | 2390.56 | 0.04030 |
| google/embeddinggemma-300m (original) | English(STS-B val) | 1500 | 0.8665 | 23.12 | 49015 | 2119.58 | 0.01542 |
| google/embeddinggemma-300m (original) | Korean(KLUE-STS val) | 519 | 0.8607 | 9.45 | 22363 | 2367.39 | 0.01820 |
| unsloth/embeddinggemma-300m-GGUF (Q8_0) | English(STS-B val) | 1500 | 0.8800 | 30.89 | 49015 | 1586.52 | 0.02060 |
| unsloth/embeddinggemma-300m-GGUF (Q8_0) | Korean(KLUE-STS val) | 519 | 0.8625 | 13.88 | 22363 | 1610.70 | 0.02675 |
| unsloth/embeddinggemma-300m-GGUF (Q4_0) | English(STS-B val) | 1500 | 0.8811 | 27.76 | 49015 | 1765.60 | 0.01851 |
| unsloth/embeddinggemma-300m-GGUF (Q4_0) | Korean(KLUE-STS val) | 519 | 0.8585 | 12.04 | 22363 | 1857.46 | 0.02320 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 (original) | English(STS-B val) | 1500 | 0.8193 | 6.75 | 52212 | 7734.65 | 0.00450 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 (original) | Korean(KLUE-STS val) | 519 | 0.7856 | 2.82 | 25317 | 8972.21 | 0.00544 |
| Xenova/distiluse-base-multilingual-cased-v2 (int8) | English(STS-B val) | 1500 | 0.7742 | 15.48 | 52212 | 3372.76 | 0.01032 |
| Xenova/distiluse-base-multilingual-cased-v2 (int8) | Korean(KLUE-STS val) | 519 | 0.7537 | 9.59 | 25317 | 2640.21 | 0.01848 |
| exp-models/dragonkue-KoEn-E5-Tiny (original) | English(STS-B val) | 1500 | 0.8718 | 12.68 | 53842 | 4244.65 | 0.00846 |
| exp-models/dragonkue-KoEn-E5-Tiny (original) | Korean(KLUE-STS val) | 519 | 0.8075 | 5.53 | 20780 | 3758.56 | 0.01065 |
| exp-models/dragonkue-KoEn-E5-Tiny-ONNX (int8) | English(STS-B val) | 1500 | 0.8684 | 15.97 | 53842 | 3371.22 | 0.01065 |
| exp-models/dragonkue-KoEn-E5-Tiny-ONNX (int8) | Korean(KLUE-STS val) | 519 | 0.7985 | 7.95 | 20780 | 2613.60 | 0.01532 |

## Retrieval

| Model | Dataset | Queries | Docs | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Elapsed(s) | Total tokens | Tokens/s | Avg sec/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jhgan/ko-sbert-sts | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.4884 | 0.6977 | 0.7907 | 0.5973 | 149.23 | 936641 | 6276.36 | 3.47054 |
| jhgan/ko-sbert-sts | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1991 | 0.3649 | 0.4692 | 0.2968 | 854.01 | 5470315 | 6405.48 | 4.04742 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.3023 | 0.3488 | 0.4186 | 0.3407 | 140.76 | 1120540 | 7960.63 | 3.27349 |
| snunlp/KR-SBERT-V40K-klueNLI-augSTS | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1564 | 0.3223 | 0.4076 | 0.2469 | 767.53 | 4825983 | 6287.68 | 3.63758 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9535 | 0.9535 | 0.8915 | 43.34 | 590622 | 13629.01 | 1.00781 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.2654 | 0.4123 | 0.5071 | 0.3520 | 292.37 | 5882605 | 20120.36 | 1.38564 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9302 | 0.9535 | 0.8857 | 156.11 | 590622 | 3783.32 | 3.63051 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (fp16) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.2322 | 0.3886 | 0.4360 | 0.3095 | 1478.05 | 5882605 | 3979.99 | 7.00496 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9302 | 0.9535 | 0.8895 | 179.31 | 590622 | 3293.81 | 4.17007 |
| Xenova/paraphrase-multilingual-MiniLM-L12-v2 (int8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1801 | 0.3412 | 0.4028 | 0.2623 | 1615.82 | 5882605 | 3640.63 | 7.65791 |
| Xenova/all-MiniLM-L12-v2 (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9302 | 1.0000 | 1.0000 | 0.9651 | 123.88 | 532160 | 4295.76 | 2.88094 |
| Xenova/all-MiniLM-L12-v2 (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.0284 | 0.0379 | 0.0521 | 0.0355 | 1431.49 | 9206787 | 6431.62 | 6.78430 |
| Xenova/all-MiniLM-L12-v2 (fp16) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9302 | 1.0000 | 1.0000 | 0.9651 | 137.06 | 532160 | 3882.71 | 3.18742 |
| Xenova/all-MiniLM-L12-v2 (fp16) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.0284 | 0.0379 | 0.0521 | 0.0355 | 1499.98 | 9206787 | 6137.95 | 7.10890 |
| Xenova/all-MiniLM-L12-v2 (int8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.9070 | 0.9767 | 0.9767 | 0.9419 | 160.93 | 532160 | 3306.76 | 3.74258 |
| Xenova/all-MiniLM-L12-v2 (int8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.0190 | 0.0379 | 0.0427 | 0.0294 | 1588.25 | -1 | 0.00 | 7.52723 |
| google/embeddinggemma-300m (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9070 | 0.9535 | 0.8775 | 332.67 | -1 | 0.00 | 7.73645 |
| google/embeddinggemma-300m (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5545 | 0.8009 | 0.8626 | 0.6764 | 4239.27 | -1 | 0.00 | 20.09134 |
| unsloth/embeddinggemma-300m-GGUF (Q8_0) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8140 | 0.9767 | 0.9767 | 0.8798 | 343.64 | -1 | 0.00 | 7.99152 |
| unsloth/embeddinggemma-300m-GGUF (Q8_0) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.6019 | 0.8104 | 0.8720 | 0.7072 | 3806.63 | -1 | 0.00 | 18.04089 |
| unsloth/embeddinggemma-300m-GGUF (Q4_0) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8605 | 0.9767 | 0.9767 | 0.9147 | 311.04 | -1 | 0.00 | 7.23360 |
| unsloth/embeddinggemma-300m-GGUF (Q4_0) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5545 | 0.7962 | 0.8531 | 0.6768 | 3486.76 | -1 | 0.00 | 16.52494 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.5116 | 0.6512 | 0.7674 | 0.5953 | 61.10 | -1 | 0.00 | 1.42102 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1469 | 0.2133 | 0.2844 | 0.1906 | 412.49 | -1 | 0.00 | 1.95494 |
| Xenova/distiluse-base-multilingual-cased-v2 (int8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.5349 | 0.6977 | 0.7674 | 0.6198 | 257.03 | -1 | 0.00 | 5.97752 |
| Xenova/distiluse-base-multilingual-cased-v2 (int8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.1517 | 0.2085 | 0.2891 | 0.1955 | 2501.73 | -1 | 0.00 | 11.85656 |
| exp-models/dragonkue-KoEn-E5-Tiny (original) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.8372 | 0.9767 | 1.0000 | 0.9039 | 140.81 | -1 | 0.00 | 3.27472 |
| exp-models/dragonkue-KoEn-E5-Tiny (original) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5640 | 0.7962 | 0.8483 | 0.6829 | 1404.78 | -1 | 0.00 | 6.65772 |
| exp-models/dragonkue-KoEn-E5-Tiny-ONNX (int8) | English(MSMARCO Passage Ranking top250 test) | 43 | 6609 | 0.7907 | 0.9767 | 0.9767 | 0.8798 | 245.45 | -1 | 0.00 | 5.70806 |
| exp-models/dragonkue-KoEn-E5-Tiny-ONNX (int8) | Korean(MIRACL-ko top250 train) | 211 | 43421 | 0.5498 | 0.7630 | 0.8483 | 0.6700 | 2581.04 | -1 | 0.00 | 12.23242 |
