#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
대규모 훈련: 1000개 데이터 × 20 에포크
- 목표: ROUGE 0.1~0.2 달성
- 검증: 매 에포크마다 5개 샘플로 평가
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
print("대규모 훈련: 한국어 vec2text")
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

# ============================================================================
# [1] 대규모 훈련 데이터 준비
# ============================================================================
print("[1/5] 훈련 데이터 준비 (1000개)")
print("-"*80)

train_texts = []
with open('output/kor_financial_pii_dataset.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        try:
            doc = json.loads(line)
            train_texts.append(doc['text'])
        except:
            continue

print(f"로드: {len(train_texts)}개 텍스트")
print(f"  샘플 1: {train_texts[0][:50]}...")
print()

# OpenAI 임베딩 생성
print("OpenAI 임베딩 생성 중...")
client = OpenAI()

batch_size = 50
embeddings = []
for i in range(0, len(train_texts), batch_size):
    batch = train_texts[i:i+batch_size]
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=batch
    )
    embeddings.extend([np.array(item.embedding, dtype=np.float32) for item in response.data])
    if (i + len(batch)) % 100 == 0:
        print(f"  [{i+len(batch)}/{len(train_texts)}] 완료")

embeddings = np.array(embeddings)
print(f"임베딩 생성 완료: {embeddings.shape}")
print()

# ============================================================================
# [2] 모델 로드
# ============================================================================
print("[2/5] 모델 로드")
print("-"*80)

model_path = "./vec_to_text_kor/vec2text/saves/ko_vec2text_1536"

t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

converter = EmbeddingToTokenConverter(1536, 768, 4).to(device)

t5_model = t5_model.to(device)

print(f"T5 params: {sum(p.numel() for p in t5_model.parameters()):,}")
print(f"Converter params: {sum(p.numel() for p in converter.parameters()):,}")
print()

# ============================================================================
# [3] 훈련 설정
# ============================================================================
print("[3/5] 훈련 설정")
print("-"*80)

optimizer = torch.optim.Adam(
    list(t5_model.parameters()) + list(converter.parameters()),
    lr=1e-4
)

num_epochs = 20
batch_size = 16
num_batches = len(train_texts) // batch_size

print(f"에포크: {num_epochs}")
print(f"배치 크기: {batch_size}")
print(f"배치 수 (에포크당): {num_batches}")
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
best_rouge = 0
no_improve_count = 0

for epoch in range(num_epochs):
    print(f"[Epoch {epoch+1}/{num_epochs}]")
    epoch_loss = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_embeddings = torch.from_numpy(embeddings[start_idx:end_idx]).to(device)
        batch_texts = train_texts[start_idx:end_idx]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        optimizer.zero_grad()

        token_embeddings = converter(batch_embeddings)

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

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f}")

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
    print()

    # ========================================================================
    # [중간 검증]
    # ========================================================================
    print(f"[Validation - Epoch {epoch+1}]")
    print("-"*40)

    t5_model.eval()
    converter.eval()

    # 5개 샘플로 평가
    test_indices = [0, 250, 500, 750, 999]
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

                status = "[OK]" if rouge_score > 0 else "[NO]"
                print(f"  Sample {idx}: ROUGE={rouge_score:.4f} {status}")

            except Exception as e:
                print(f"  Sample {idx}: ERROR")

    if epoch_rouge_scores:
        avg_rouge = np.mean(epoch_rouge_scores)
        print(f"\n  Average ROUGE-1: {avg_rouge:.4f}")

        if avg_rouge > best_rouge:
            best_rouge = avg_rouge
            no_improve_count = 0
            print(f"  [IMPROVED] New best: {best_rouge:.4f}")
        else:
            no_improve_count += 1
            print(f"  [NO IMPROVE] {no_improve_count} epochs without improvement")

        # Early stopping
        if no_improve_count >= 5:
            print(f"\n  ** Early stopping at epoch {epoch+1}")
            break

    print()

    t5_model.train()
    converter.train()

print()

# ============================================================================
# [5] 최종 모델 저장
# ============================================================================
print("[5/5] 최종 모델 저장")
print("-"*80)

output_path = "./vec_to_text_kor/vec2text/saves/ko_vec2text_1536_v3_large"
os.makedirs(output_path, exist_ok=True)

t5_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

converter_path = os.path.join(output_path, "embedding_converter.pt")
torch.save({
    'state_dict': converter.state_dict(),
    'config': {
        'embedding_dim': 1536,
        'hidden_dim': 768,
        'num_tokens': 4
    }
}, converter_path)

print(f"Model saved: {output_path}")
print()

# ============================================================================
# [최종 검증]
# ============================================================================
print("[Final Validation]")
print("-"*80)

t5_model.eval()
converter.eval()

test_indices = [0, 1, 2, 3, 4]
with torch.no_grad():
    rouge_scores = []
    for idx in test_indices:
        embedding = torch.from_numpy(embeddings[idx]).to(device).unsqueeze(0)
        token_emb = converter(embedding)

        outputs = t5_model.generate(
            inputs_embeds=token_emb,
            max_length=256,
            num_beams=1
        )

        restored = tokenizer.decode(outputs[0], skip_special_tokens=True)
        original = train_texts[idx][:100]
        rouge_score = rouge.score(original, restored)['rouge1'].fmeasure
        rouge_scores.append(rouge_score)

        print(f"Sample {idx}: ROUGE={rouge_score:.4f}")

    avg_final = np.mean(rouge_scores)
    print(f"\nAverage ROUGE-1: {avg_final:.4f}")

    if avg_final > 0.2:
        print("[EXCELLENT] Good restoration quality")
    elif avg_final > 0.1:
        print("[GOOD] Acceptable quality")
    else:
        print("[WEAK] Need more improvement")

print()
print("="*80)
print("Training Complete!")
print("="*80)

