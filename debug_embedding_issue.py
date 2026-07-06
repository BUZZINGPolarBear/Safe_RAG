#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
디버깅: T5 입력 임베딩의 범위와 크기 확인
"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

print("="*80)
print("디버깅: T5 입력 임베딩 분석")
print("="*80)
print()

# T5 모델 로드
model_path = "./vec_to_text_kor/vec2text/saves/ko_vec2text_1536"
device = "cuda" if torch.cuda.is_available() else "cpu"

t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
t5_model = t5_model.to(device)

print("[T5 모델 입력 임베딩 분석]")
print("-"*80)

# 입력 임베딩 테이블 확인
shared_embeddings = t5_model.shared
print(f"공유 임베딩 테이블:")
print(f"  크기: {shared_embeddings.weight.shape}")  # (vocab_size, d_model)
print(f"  어휘 수: {shared_embeddings.num_embeddings}")
print(f"  임베딩 차원: {shared_embeddings.embedding_dim}")
print()

# 엔코더 입력 임베딩
encoder_embed = t5_model.encoder.embed_tokens
print(f"인코더 입력 임베딩:")
print(f"  크기: {encoder_embed.weight.shape}")
print()

# 범위 확인
with torch.no_grad():
    print("[임베딩 가중치 통계]")
    print(f"  Min: {shared_embeddings.weight.min():.4f}")
    print(f"  Max: {shared_embeddings.weight.max():.4f}")
    print(f"  Mean: {shared_embeddings.weight.mean():.4f}")
    print(f"  Std: {shared_embeddings.weight.std():.4f}")
print()

# 정상적인 입력 토큰과 비교
print("[정상 입력 토큰 비교]")
print("-"*80)

test_text = "안녕하세요"
inputs = tokenizer(test_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

print(f"테스트 텍스트: '{test_text}'")
print(f"토큰 IDs: {input_ids.tolist()}")

# 토큰 임베딩 얻기
with torch.no_grad():
    token_embeddings = t5_model.encoder.embed_tokens(input_ids)

print(f"토큰 임베딩 shape: {token_embeddings.shape}")
print(f"토큰 임베딩 범위:")
print(f"  Min: {token_embeddings.min():.4f}")
print(f"  Max: {token_embeddings.max():.4f}")
print(f"  Mean: {token_embeddings.mean():.4f}")
print(f"  Std: {token_embeddings.std():.4f}")
print()

# 우리가 생성하는 임베딩과 비교
print("[우리 변환 레이어 출력 vs 정상 토큰 임베딩]")
print("-"*80)

from embedding_to_token_converter import EmbeddingToTokenConverter

converter = EmbeddingToTokenConverter(1536, 768, 4).to(device)

# 더미 임베딩
dummy_embedding = torch.randn(1, 1536).to(device)
with torch.no_grad():
    our_output = converter(dummy_embedding)

print(f"우리 출력 shape: {our_output.shape}")
print(f"우리 출력 범위:")
print(f"  Min: {our_output.min():.4f}")
print(f"  Max: {our_output.max():.4f}")
print(f"  Mean: {our_output.mean():.4f}")
print(f"  Std: {our_output.std():.4f}")
print()

# 문제 진단
print("[문제 진단]")
print("-"*80)

if our_output.std().item() > token_embeddings.std().item() * 2:
    print("⚠ 문제: 우리 출력의 분산이 너무 큼")
    print("→ 정규화 추가 필요")

if our_output.mean().abs().item() > token_embeddings.mean().abs().item() * 2:
    print("⚠ 문제: 우리 출력의 평균이 0에서 너무 멈")
    print("→ 평균 센터링 필요")

print()
print("="*80)
print("해결책: 변환 레이어에 정규화 추가")
print("="*80)

