#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Option A: 변환 레이어 + T5 함께 재훈련
- 임베딩 → 텍스트 복원 목표
- 중간마다 검증으로 진행 상황 확인
"""

import json
import torch
import torch.nn as nn
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from openai import OpenAI
import os
from rouge_score import rouge_scorer
from embedding_to_token_converter import EmbeddingToTokenConverter

print("="*80)
print("Option A: 변환 레이어 + T5 함께 재훈련")
print("="*80)
print()

# ============================================================================
# [Setup] API 키 설정
# ============================================================================
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

# ============================================================================
# [1] 훈련 데이터 준비
# ============================================================================
print("[1/5] 훈련 데이터 준비 (임베딩 + 텍스트 페어)")
print("-"*80)

# 한국어 데이터셋 로드
train_texts = []
with open('output/kor_financial_pii_dataset.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 100:  # 100개 샘플로 훈련
            break
        try:
            doc = json.loads(line)
            train_texts.append(doc['text'])
        except:
            continue

print(f"로드: {len(train_texts)}개 텍스트")
print(f"  샘플: {train_texts[0][:50]}...")
print()

# OpenAI 임베딩 생성
print("OpenAI 임베딩 생성 중...")
client = OpenAI()

batch_size = 20
embeddings = []
for i in range(0, len(train_texts), batch_size):
    batch = train_texts[i:i+batch_size]
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=batch
    )
    embeddings.extend([np.array(item.embedding, dtype=np.float32) for item in response.data])
    print(f"  [{i+len(batch)}/{len(train_texts)}] 생성 완료")

embeddings = np.array(embeddings)
print(f"임베딩 생성 완료: {embeddings.shape}")
print()

# ============================================================================
# [2] 모델 로드
# ============================================================================
print("[2/5] 모델 로드 및 초기화")
print("-"*80)

model_path = "./vec_to_text_kor/vec2text/saves/ko_vec2text_1536"
device = "cuda" if torch.cuda.is_available() else "cpu"

t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

converter = EmbeddingToTokenConverter(1536, 768, 4).to(device)

t5_model = t5_model.to(device)

print(f"T5 모델: {sum(p.numel() for p in t5_model.parameters()):,} 파라미터")
print(f"변환 레이어: {sum(p.numel() for p in converter.parameters()):,} 파라미터")
print()

# ============================================================================
# [3] 훈련 설정
# ============================================================================
print("[3/5] 훈련 설정")
print("-"*80)

# 토크나이저 설정
for text in train_texts[:1]:
    inputs = tokenizer(text[:100], return_tensors="pt", max_length=128, truncation=True)
    print(f"입력 토큰 길이: {inputs['input_ids'].shape[1]}")

# 최적화기
optimizer = torch.optim.Adam(
    list(t5_model.parameters()) + list(converter.parameters()),
    lr=1e-4
)

num_epochs = 5
batch_size = 8
print(f"훈련 설정:")
print(f"  에포크: {num_epochs}")
print(f"  배치: {batch_size}")
print(f"  샘플: {len(train_texts)}")
print()

# ============================================================================
# [4] 훈련 루프
# ============================================================================
print("[4/5] 훈련 진행")
print("-"*80)
print()

t5_model.train()
converter.train()

all_losses = []

for epoch in range(num_epochs):
    print(f"[Epoch {epoch+1}/{num_epochs}]")
    epoch_loss = 0
    num_batches = len(train_texts) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        # 배치 데이터
        batch_embeddings = torch.from_numpy(embeddings[start_idx:end_idx]).to(device)
        batch_texts = train_texts[start_idx:end_idx]

        # 텍스트 토크나이징
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        optimizer.zero_grad()

        # 임베딩 → 토큰 형태
        token_embeddings = converter(batch_embeddings)  # (batch, 4, 768)

        # T5에 입력
        outputs = t5_model(
            inputs_embeds=token_embeddings,
            labels=inputs["input_ids"]
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(t5_model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        all_losses.append(loss.item())

        if (batch_idx + 1) % 3 == 0:
            print(f"  [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f}")

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} 평균 손실: {avg_epoch_loss:.4f}")
    print()

    # ============================================================================
    # [중간 검증] 각 에포크마다 복원 품질 확인
    # ============================================================================
    print(f"[중간 검증 - Epoch {epoch+1}]")
    print("-"*40)

    t5_model.eval()
    converter.eval()

    # 테스트 샘플 (처음 3개)
    test_indices = [0, 25, 50]
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
    epoch_rouge_scores = []

    with torch.no_grad():
        for idx in test_indices:
            try:
                embedding = torch.from_numpy(embeddings[idx]).to(device).unsqueeze(0)
                token_emb = converter(embedding)

                outputs = t5_model.generate(
                    inputs_embeds=token_emb,
                    max_length=256,
                    num_beams=1,
                    do_sample=False
                )

                restored = tokenizer.decode(outputs[0], skip_special_tokens=True)
                original = train_texts[idx][:100]

                rouge_score = rouge.score(original, restored)['rouge1'].fmeasure
                epoch_rouge_scores.append(rouge_score)

                # 결과 출력
                status = "✓" if len(restored) > 0 and rouge_score > 0 else "✗"
                print(f"  샘플 {idx}: ROUGE={rouge_score:.4f} {status}")
                if len(restored) < 50:
                    print(f"    복원: {restored if restored else '(빈 텍스트)'}")
                else:
                    print(f"    복원: {restored[:50]}...")

            except Exception as e:
                print(f"  샘플 {idx}: 오류 - {str(e)[:40]}")

    if epoch_rouge_scores:
        avg_rouge = np.mean(epoch_rouge_scores)
        print(f"\n  평균 ROUGE-1: {avg_rouge:.4f}")

        if avg_rouge > 0.3:
            print(f"  STATUS: 우수! (목표 달성)")
        elif avg_rouge > 0.1:
            print(f"  STATUS: 개선 중...")
        elif avg_rouge > 0:
            print(f"  STATUS: 텍스트 생성 시작")
        else:
            print(f"  STATUS: 아직 미약")

    print()

    t5_model.train()
    converter.train()

print()

# ============================================================================
# [5] 최종 모델 저장
# ============================================================================
print("[5/5] 최종 모델 저장")
print("-"*80)

output_path = "./vec_to_text_kor/vec2text/saves/ko_vec2text_1536_v2_retrained"
os.makedirs(output_path, exist_ok=True)

# 모델 저장
t5_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# 변환 레이어 저장
converter_path = os.path.join(output_path, "embedding_converter.pt")
torch.save({
    'state_dict': converter.state_dict(),
    'config': {
        'embedding_dim': 1536,
        'hidden_dim': 768,
        'num_tokens': 4
    }
}, converter_path)

print(f"모델 저장 완료: {output_path}")
print()

# 최종 검증
print("[최종 검증]")
print("-"*80)

t5_model.eval()
converter.eval()

test_embedding = torch.from_numpy(embeddings[0]).to(device).unsqueeze(0)
with torch.no_grad():
    token_emb = converter(test_embedding)
    outputs = t5_model.generate(
        inputs_embeds=token_emb,
        max_length=256,
        num_beams=1
    )

final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"최종 생성 샘플:")
print(f"  원본: {train_texts[0][:60]}...")
print(f"  복원: {final_text[:60]}..." if final_text else "  복원: (빈 텍스트)")

print()
print("="*80)
print("재훈련 완료!")
print("="*80)

