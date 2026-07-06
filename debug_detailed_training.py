#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
더 정밀한 디버깅: 훈련 중 모델의 실제 동작 확인
- 변환 레이어 출력 값
- T5 어텐션 가중치
- 생성된 토큰
"""

import json
import torch
import torch.nn as nn
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from openai import OpenAI
import os
from embedding_to_token_converter import EmbeddingToTokenConverter

print("="*80)
print("디버깅: 훈련 중 모델 동작 확인")
print("="*80)
print()

# ============================================================================
# [Setup]
# ============================================================================
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

# ============================================================================
# [1] 데이터 준비
# ============================================================================
print("[1/4] 데이터 준비")
print("-"*80)

train_texts = []
with open('output/kor_financial_pii_dataset.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:  # 5개만
            break
        try:
            doc = json.loads(line)
            train_texts.append(doc['text'])
        except:
            continue

print(f"로드: {len(train_texts)}개 텍스트")
for i, text in enumerate(train_texts):
    print(f"  [{i}] {text[:40]}...")
print()

# OpenAI 임베딩
client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=train_texts
)
embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
print(f"임베딩: {embeddings.shape}")
print()

# ============================================================================
# [2] 모델 로드
# ============================================================================
print("[2/4] 모델 로드")
print("-"*80)

model_path = "./vec_to_text_kor/vec2text/saves/ko_vec2text_1536"
device = "cuda" if torch.cuda.is_available() else "cpu"

t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
converter = EmbeddingToTokenConverter(1536, 768, 4).to(device)

t5_model = t5_model.to(device)
t5_model.eval()
converter.eval()

print("로드 완료")
print()

# ============================================================================
# [3] 한 번의 Forward Pass 상세 분석
# ============================================================================
print("[3/4] Forward Pass 상세 분석")
print("-"*80)

idx = 0
embedding = torch.from_numpy(embeddings[idx]).to(device).unsqueeze(0)
original_text = train_texts[idx]

print(f"원본: {original_text[:50]}...")
print()

# Step 1: 변환 레이어
print("[Step 1] 변환 레이어 출력")
with torch.no_grad():
    token_emb = converter(embedding)

print(f"  입력 임베딩: {embedding.shape}, range=[{embedding.min():.2f}, {embedding.max():.2f}]")
print(f"  출력 임베딩: {token_emb.shape}, range=[{token_emb.min():.2f}, {token_emb.max():.2f}]")
print(f"  출력 평균: {token_emb.mean():.4f}, 표준편차: {token_emb.std():.4f}")
print()

# Step 2: T5 인코더 분석
print("[Step 2] T5 인코더 동작 확인")
with torch.no_grad():
    # 인코더 입출력
    encoder = t5_model.encoder
    encoder_output = encoder(inputs_embeds=token_emb)

print(f"  인코더 출력: {encoder_output.last_hidden_state.shape}")
print(f"  범위: [{encoder_output.last_hidden_state.min():.2f}, {encoder_output.last_hidden_state.max():.2f}]")
print(f"  평균: {encoder_output.last_hidden_state.mean():.4f}")
print()

# Step 3: 생성 과정 추적
print("[Step 3] 생성 과정 추적")
print("-"*40)

with torch.no_grad():
    # 토크나이저를 위한 더미 input_ids
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 1

    # 생성 (최소한)
    outputs = t5_model.generate(
        inputs_embeds=token_emb,
        max_length=64,
        num_beams=1,
        do_sample=False,
        output_scores=True,  # 스코어 반환
        return_dict_in_generate=True  # 상세 반환
    )

print(f"생성된 토큰 ID: {outputs.sequences[0]}")
print(f"생성 길이: {len(outputs.sequences[0])}")

# 디코딩
generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print(f"생성된 텍스트: '{generated_text}'")
print(f"텍스트 길이: {len(generated_text)}")
print()

# ============================================================================
# [4] 정상 토큰 vs 우리 토큰 비교
# ============================================================================
print("[4/4] 정상 토큰 vs 우리 토큰 비교")
print("-"*80)

# 정상 입력 처리
inputs = tokenizer(original_text[:100], return_tensors="pt", max_length=64, truncation=True)
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    # 정상 토큰 임베딩
    normal_token_emb = t5_model.encoder.embed_tokens(input_ids)

    # 정상 인코더
    normal_encoder_output = t5_model.encoder(input_ids=input_ids)

    # 정상 생성 (원본 텍스트로)
    normal_output = t5_model.generate(
        input_ids=input_ids,
        max_length=64,
        num_beams=1,
        do_sample=False
    )

normal_text = tokenizer.decode(normal_output[0], skip_special_tokens=True)

print("[정상 토큰으로 처리]")
print(f"  토큰 임베딩: {normal_token_emb.shape}, range=[{normal_token_emb.min():.2f}, {normal_token_emb.max():.2f}]")
print(f"  인코더 출력: {normal_encoder_output.last_hidden_state.shape}")
print(f"  생성: '{normal_text}'")
print()

print("[우리 토큰으로 처리]")
print(f"  토큰 임베딩: {token_emb.shape}, range=[{token_emb.min():.2f}, {token_emb.max():.2f}]")
print(f"  인코더 출력: {encoder_output.last_hidden_state.shape}")
print(f"  생성: '{generated_text}'")
print()

# ============================================================================
# [문제 진단]
# ============================================================================
print("[진단]")
print("-"*80)

if len(generated_text) == 0:
    print("⚠ 문제: 생성된 텍스트가 완전히 비어있음")
    print("  → 디코더가 작동하지 않음")
elif "." in generated_text and generated_text.count(".") > len(generated_text) / 2:
    print("⚠ 문제: 점 문자만 반복 생성")
    print("  → 디코더가 혼란스러워함, 특수 토큰으로 반응")
elif len(generated_text) > 0 and generated_text != normal_text:
    print(f"⚠ 문제: 텍스트 생성되지만 원본과 다름")
    print(f"  - 정상: '{normal_text}'")
    print(f"  - 우리: '{generated_text}'")
    print("  → 모델이 임베딩을 올바르게 이해하지 못함")
else:
    print("✓ 정상 작동")

print()
print("="*80)

