# Multimodal Embedding Benchmark

## STS

| Model | Dataset | Samples | Spearman | Elapsed(s) | Avg sec/pair |
|---|---|---:|---:|---:|---:|
| google/embeddinggemma-2 (original) | English(STS-B val) | 1500 | 0.8470 | 2.68 | 0.00179 |
| google/embeddinggemma-2 (original) | Korean(KLUE-STS val) | 519 | 0.8201 | 1.23 | 0.00237 |
| Qwen/Qwen3-VL-Embedding-2B (original) | English(STS-B val) | 1500 | 0.8696 | 9.47 | 0.00631 |
| Qwen/Qwen3-VL-Embedding-2B (original) | Korean(KLUE-STS val) | 519 | 0.8637 | 5.28 | 0.01018 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | English(STS-B val) | 1500 | 0.6275 | 151.12 | 0.10075 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | Korean(KLUE-STS val) | 519 | 0.7282 | 60.37 | 0.11632 |
| gemini-embedding-2 (API) | English(STS-B val) | 1500 | 0.7928 | 63.05 | 0.04203 |
| gemini-embedding-2 (API) | Korean(KLUE-STS val) | 519 | 0.7418 | 24.48 | 0.04717 |
| voyage-multimodal-3.5 (API) | English(STS-B val) | 1500 | 0.8983 | 44.18 | 0.02945 |
| voyage-multimodal-3.5 (API) | Korean(KLUE-STS val) | 519 | 0.8379 | 24.20 | 0.04663 |

## Retrieval

| Model | Dataset | Queries | Corpus | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | Avg sec/text | Avg sec/image |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| google/embeddinggemma-2 (original) | English(XM3600 T2I) | 7200 | 3600 | 0.5936 | 0.8282 | 0.8868 | 0.6931 | 0.7211 | 0.00072 | 0.04655 |
| google/embeddinggemma-2 (original) | English(XM3600 I2T) | 3600 | 7200 | 0.6708 | 0.8931 | 0.9456 | 0.7645 | 0.6892 | 0.00072 | 0.04655 |
| google/embeddinggemma-2 (original) | Korean(XM3600 T2I) | 7650 | 3600 | 0.6229 | 0.8498 | 0.9044 | 0.7202 | 0.7472 | 0.00093 | 0.04655 |
| google/embeddinggemma-2 (original) | Korean(XM3600 I2T) | 3600 | 7650 | 0.7133 | 0.9194 | 0.9572 | 0.8011 | 0.7187 | 0.00093 | 0.04655 |
| google/embeddinggemma-2 (original) | English(ViDoRe DocVQA test) | 451 | 500 | 0.2993 | 0.4545 | 0.5410 | 0.3695 | 0.3773 | 0.00085 | 0.05100 |
| google/embeddinggemma-2 (original) | Korean(KoViDoRe v2 economic) | 163 | 1477 | 0.0798 | 0.2577 | 0.4049 | 0.1616 | 0.0980 | 0.00137 | 0.05221 |
| Qwen/Qwen3-VL-Embedding-2B (original) | English(XM3600 T2I) | 7200 | 3600 | 0.5962 | 0.8311 | 0.8915 | 0.6958 | 0.7235 | 0.00315 | 0.05169 |
| Qwen/Qwen3-VL-Embedding-2B (original) | English(XM3600 I2T) | 3600 | 7200 | 0.6814 | 0.9044 | 0.9506 | 0.7770 | 0.7007 | 0.00315 | 0.05169 |
| Qwen/Qwen3-VL-Embedding-2B (original) | Korean(XM3600 T2I) | 7650 | 3600 | 0.6638 | 0.8861 | 0.9325 | 0.7580 | 0.7855 | 0.00437 | 0.05169 |
| Qwen/Qwen3-VL-Embedding-2B (original) | Korean(XM3600 I2T) | 3600 | 7650 | 0.7469 | 0.9328 | 0.9694 | 0.8254 | 0.7446 | 0.00437 | 0.05169 |
| Qwen/Qwen3-VL-Embedding-2B (original) | English(ViDoRe DocVQA test) | 451 | 500 | 0.4235 | 0.5676 | 0.6297 | 0.4907 | 0.5017 | 0.00345 | 0.20419 |
| Qwen/Qwen3-VL-Embedding-2B (original) | Korean(KoViDoRe v2 economic) | 163 | 1477 | 0.0982 | 0.2515 | 0.3804 | 0.1714 | 0.1061 | 0.00614 | 0.19109 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | English(XM3600 T2I) | 7200 | 3600 | 0.4633 | 0.7221 | 0.8044 | 0.5727 | 0.6017 | 0.05211 | 0.83169 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | English(XM3600 I2T) | 3600 | 7200 | 0.5053 | 0.7564 | 0.8447 | 0.6132 | 0.5193 | 0.05211 | 0.83169 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | Korean(XM3600 T2I) | 7650 | 3600 | 0.3929 | 0.6495 | 0.7489 | 0.5028 | 0.5294 | 0.05177 | 0.83169 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | Korean(XM3600 I2T) | 3600 | 7650 | 0.3464 | 0.5864 | 0.6900 | 0.4500 | 0.3512 | 0.05177 | 0.83169 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | English(ViDoRe DocVQA test) | 451 | 500 | 0.4612 | 0.6674 | 0.7295 | 0.5458 | 0.5686 | 0.05072 | 0.81254 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | Korean(KoViDoRe v2 economic) | 163 | 1477 | 0.0245 | 0.0920 | 0.1411 | 0.0523 | 0.0328 | 0.05165 | 0.71157 |
| gemini-embedding-2 (API) | English(XM3600 T2I) | 7200 | 3600 | 0.5944 | 0.8214 | 0.8831 | 0.6908 | 0.7174 | 0.05637 | 0.18082 |
| gemini-embedding-2 (API) | English(XM3600 I2T) | 3600 | 7200 | 0.6858 | 0.8961 | 0.9475 | 0.7765 | 0.6897 | 0.05637 | 0.18082 |
| gemini-embedding-2 (API) | Korean(XM3600 T2I) | 7650 | 3600 | 0.7267 | 0.9148 | 0.9488 | 0.8062 | 0.8301 | 0.02199 | 0.18082 |
| gemini-embedding-2 (API) | Korean(XM3600 I2T) | 3600 | 7650 | 0.8283 | 0.9661 | 0.9864 | 0.8871 | 0.8273 | 0.02199 | 0.18082 |
| gemini-embedding-2 (API) | English(ViDoRe DocVQA test) | 451 | 500 | 0.2195 | 0.3636 | 0.4390 | 0.2838 | 0.2933 | 0.02438 | 0.24323 |
| gemini-embedding-2 (API) | Korean(KoViDoRe v2 economic) | 163 | 1477 | 0.1472 | 0.3129 | 0.4847 | 0.2306 | 0.1429 | 0.02432 | 0.22189 |
| voyage-multimodal-3.5 (API) | English(XM3600 T2I) | 7200 | 3600 | 0.5875 | 0.8300 | 0.8896 | 0.6909 | 0.7197 | 0.01358 | 0.26699 |
| voyage-multimodal-3.5 (API) | English(XM3600 I2T) | 3600 | 7200 | 0.6883 | 0.9022 | 0.9483 | 0.7786 | 0.6987 | 0.01358 | 0.26699 |
| voyage-multimodal-3.5 (API) | Korean(XM3600 T2I) | 7650 | 3600 | 0.6213 | 0.8520 | 0.9090 | 0.7190 | 0.7466 | 0.01893 | 0.26699 |
| voyage-multimodal-3.5 (API) | Korean(XM3600 I2T) | 3600 | 7650 | 0.6994 | 0.9006 | 0.9489 | 0.7864 | 0.6951 | 0.01893 | 0.26699 |
| voyage-multimodal-3.5 (API) | English(ViDoRe DocVQA test) | 451 | 500 | 0.4789 | 0.6563 | 0.7317 | 0.5555 | 0.5684 | 0.01742 | 0.57340 |
| voyage-multimodal-3.5 (API) | Korean(KoViDoRe v2 economic) | 163 | 1477 | 0.2331 | 0.4908 | 0.5951 | 0.3402 | 0.2394 | 0.02783 | 0.55770 |

## Cost (USD)

| Model | Cost |
|---|---:|
| google/embeddinggemma-2 (original) | 0.0000 |
| Qwen/Qwen3-VL-Embedding-2B (original) | 0.0000 |
| nvidia/llama-nemotron-embed-vl-1b-v2:free (OpenRouter) | 0.0000 |
| gemini-embedding-2 (API) | 0.7446 |
| voyage-multimodal-3.5 (API) | 2.1170 |
| **Total** | **2.8616** |

## 참고

- 로컬 모델(google/embeddinggemma-2, Qwen/Qwen3-VL-Embedding-2B)은 RTX 3090 에서 bfloat16, sentence-transformers 로 돌렸습니다.
- nvidia/llama-nemotron-embed-vl-1b-v2 는 OpenRouter 무료 엔드포인트(provider: Nvidia)로 돌렸습니다. 텍스트는 `input_type` (retrieval query=`query`, STS=`passage`)으로 모델의 `query: `/`passage: ` prefix 를 서버가 붙이고, 이미지는 `input_type` 없이 보냈습니다 (로컬 HF 모델의 document 임베딩과 cos 0.99 로 일치함을 확인).
- gemini-embedding-2 는 Gemini API(google-genai), voyage-multimodal-3.5 는 Voyage API 로 돌렸습니다. 둘 다 기본 출력 차원(3072 / 1024)입니다.
- 모든 이미지는 긴 변 1280px 이하로 줄이고 JPEG(q90)로 다시 인코딩해 모든 모델에 같은 픽셀을 넣었습니다 (XM3600 은 원본이 640px 이하라 리사이즈 없음). 그 뒤의 해상도/타일링은 각 모델의 기본값입니다 (EmbeddingGemma 2 는 이미지당 280 토큰).
- query prefix: EmbeddingGemma 2 와 Gemini 는 `task: search result | query: `, Qwen3-VL-Embedding 은 instruction 으로 XM3600 에 `Find me an everyday image that matches the given caption.`, 문서 검색에 `Find a document page image that answers the user's question.`, Voyage 는 `input_type=query`/`document` 를 썼습니다. 이미지에는 prefix 를 붙이지 않았습니다. STS 는 이전 글과 같이 prefix 없이 원문을 넣었습니다 (Nemotron 은 서버가 prefix 를 항상 붙이므로 `passage`).
- XM3600 은 이미지 3,600장을 영어/한국어 캡션과 짝지은 데이터셋이고, I2T 는 같은 임베딩으로 이미지→캡션 방향을 계산한 값입니다 (그 이미지의 캡션 중 하나라도 top-k 에 있으면 hit, 캡션은 query prompt 로 인코딩).
- Recall@k 는 정답이 top-k 안에 하나라도 있는 비율, nDCG@5 는 binary relevance 기준입니다.
- 인코딩 시간은 배치별 소요 시간의 합(재시도/대기 포함)이고, API 는 병렬 요청(OpenRouter 2, Gemini/Voyage 4)이었으므로 실제 경과 시간은 이보다 짧습니다. 로컬 모델은 같은 GPU 에서 다른 작업이 함께 돌던 상태라 참고용입니다.
- 비용(USD)은 Voyage 는 응답 usage 와 공시 단가, Gemini 는 usage 를 주지 않아 공시 단가(이미지당 $0.00012, 텍스트 $0.20/1M, 토큰 수는 Gemma 토크나이저로 추정)로 계산했습니다. Voyage 계정 무료 할당량이 적용되면 실제 청구액은 더 적습니다. 위 표 외에 스모크 테스트와 아래 진단으로 약 $0.07 를 더 썼습니다.
- Gemini 의 문서 검색 점수가 낮은 것이 1280px 리사이즈 때문인지 보려고 ViDoRe DocVQA 를 원본 해상도로 다시 돌려 봤는데, Recall@1 0.2195 → 0.2439, nDCG@5 0.2933 → 0.3157 로 차이가 작았습니다.
- STS 에 prefix 를 넣지 않았기 때문에, `task: sentence similarity | query: ` 같은 prefix 를 권장하는 EmbeddingGemma 2 / Gemini 는 STS 점수가 실제 권장 사용법보다 낮게 나왔을 수 있습니다.
