#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
재훈련된 한국어 vec2text 모델 최종 테스트
- 모델: ko_vec2text_1536_v2_retrained
- 데이터: 한국어 PII 데이터셋 (10개 샘플)
- 평가: ROUGE 점수, 복원 품질
"""

import json
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from openai import OpenAI
import os
from rouge_score import rouge_scorer

print("="*80)
print("재훈련된 한국어 vec2text 모델 최종 테스트")
print("="*80)
print()

# ============================================================================
# [Setup]
# ============================================================================
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# ============================================================================
# [1] 테스트 데이터 준비
# ============================================================================
print("[1/4] 테스트 데이터 준비")
print("-"*80)

test_texts = []
with open('output/kor_financial_pii_dataset.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        try:
            doc = json.loads(line)
            test_texts.append(doc['text'])
        except:
            continue

print(f"로드: {len(test_texts)}개 텍스트")
for i, text in enumerate(test_texts[:3]):
    print(f"  [{i}] {text[:50]}...")
print()

# ============================================================================
# [2] OpenAI 임베딩 생성
# ============================================================================
print("[2/4] OpenAI 임베딩 생성")
print("-"*80)

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=test_texts
)

embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
print(f"생성: {embeddings.shape}")
print()

# ============================================================================
# [3] 재훈련된 모델 로드
# ============================================================================
print("[3/4] 재훈련된 모델 로드")
print("-"*80)

model_path = "./vec_to_text_kor/vec2text/saves/ko_vec2text_1536_v2_retrained"

try:
    t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 변환 레이어 로드
    from embedding_to_token_converter import EmbeddingToTokenConverter
    converter = EmbeddingToTokenConverter(1536, 768, 4).to(device)

    converter_state = torch.load(f"{model_path}/embedding_converter.pt", map_location=device)
    converter.load_state_dict(converter_state['state_dict'])
    converter.eval()

    t5_model = t5_model.to(device)
    t5_model.eval()

    print("OK: Model loaded: " + model_path)
    print(f"  T5 params: {sum(p.numel() for p in t5_model.parameters()):,}")
    print(f"  Converter params: {sum(p.numel() for p in converter.parameters()):,}")

except Exception as e:
    print("ERROR: " + str(e)[:100])
    exit(1)

print()

# ============================================================================
# [4] 텍스트 복원 및 평가
# ============================================================================
print("[4/4] 텍스트 복원 및 평가")
print("-"*80)
print()

rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
results = []

with torch.no_grad():
    for i, (original_text, embedding) in enumerate(zip(test_texts, embeddings)):
        try:
            # 임베딩 → 토큰 형태
            embedding_tensor = torch.from_numpy(embedding).to(device).unsqueeze(0)
            token_embeddings = converter(embedding_tensor)

            # 텍스트 생성
            outputs = t5_model.generate(
                inputs_embeds=token_embeddings,
                max_length=256,
                num_beams=1,
                do_sample=False
            )

            restored_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 평가
            original_for_eval = original_text[:100]
            rouge1 = rouge.score(original_for_eval, restored_text)['rouge1'].fmeasure
            rougeL = rouge.score(original_for_eval, restored_text)['rougeL'].fmeasure

            results.append({
                'idx': i,
                'original': original_text[:60],
                'restored': restored_text[:60] if restored_text else '(빈 텍스트)',
                'rouge1': rouge1,
                'rougeL': rougeL,
                'restored_len': len(restored_text),
                'success': len(restored_text) > 0
            })

            print(f"[Sample {i}]")
            print(f"  Original: {original_text[:50]}...")
            print(f"  Restored: {restored_text[:50] if restored_text else '(empty)'}...")
            print(f"  ROUGE-1: {rouge1:.4f} | ROUGE-L: {rougeL:.4f}")
            print()

        except Exception as e:
            print(f"[샘플 {i}] 오류: {str(e)[:50]}")
            results.append({
                'idx': i,
                'original': original_text[:60],
                'restored': '',
                'rouge1': 0,
                'rougeL': 0,
                'restored_len': 0,
                'success': False
            })
            print()

# ============================================================================
# [결과 분석]
# ============================================================================
print("="*80)
print("[최종 평가]")
print("="*80)
print()

success_count = sum(1 for r in results if r['success'])
valid_results = [r for r in results if r['success']]

print(f"성공 여부:")
print(f"  전체: {len(results)}개")
print(f"  성공: {success_count}개 ({100*success_count/len(results):.1f}%)")
print(f"  비어있음: {len(results) - success_count}개")
print()

if valid_results:
    avg_rouge1 = np.mean([r['rouge1'] for r in valid_results])
    avg_rougeL = np.mean([r['rougeL'] for r in valid_results])
    max_rouge1 = max([r['rouge1'] for r in valid_results])
    min_rouge1 = min([r['rouge1'] for r in valid_results])

    print(f"ROUGE 점수 (성공한 샘플만):")
    print(f"  ROUGE-1: 평균={avg_rouge1:.4f}, 최대={max_rouge1:.4f}, 최소={min_rouge1:.4f}")
    print(f"  ROUGE-L: 평균={avg_rougeL:.4f}")
    print()

print("[상태 판정]")
print("-"*80)

if success_count == 0:
    print("[FAIL] Complete failure - all samples produced empty text")
    print("  -> Recheck architecture")
elif success_count < len(results) / 2:
    print("[PARTIAL] Partial success")
    print(f"  -> {success_count}/{len(results)} samples generated text")
    if valid_results:
        print(f"  -> Average ROUGE: {avg_rouge1:.4f}")
    print("  -> More optimization needed")
elif avg_rouge1 < 0.1:
    print("[WEAK] Generation started but quality very low")
    print(f"  -> Average ROUGE: {avg_rouge1:.4f}")
    print("  -> Need more data + epochs")
elif avg_rouge1 < 0.3:
    print("[IMPROVING] Making progress")
    print(f"  -> Average ROUGE: {avg_rouge1:.4f}")
    print("  -> Can improve with more training")
else:
    print("[GOOD] Acceptable quality")
    print(f"  -> Average ROUGE: {avg_rouge1:.4f}")
    print("  -> Practical level achieved")

print()
print("="*80)

