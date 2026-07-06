#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
최종 검증: 훈련된 모델과 노트북 통합 테스트
훈련 완료 후 실행 가능
"""

import json
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from openai import OpenAI
import os
from rouge_score import rouge_scorer
from embedding_to_token_converter import EmbeddingToTokenConverter

def test_model(model_path, num_samples=20):
    """모델 테스트"""

    print("="*80)
    print(f"Final Validation: {model_path}")
    print("="*80)
    print()

    # Setup
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('OPENAI_API_KEY='):
                os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    test_texts = []
    with open('output/kor_financial_pii_dataset.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                doc = json.loads(line)
                test_texts.append(doc['text'])
            except:
                continue

    print(f"[1/4] Load {len(test_texts)} test samples")
    print()

    # Generate embeddings
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=test_texts
    )
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)

    print(f"[2/4] OpenAI embeddings: {embeddings.shape}")
    print()

    # Load model
    try:
        t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        converter = EmbeddingToTokenConverter(1536, 768, 4).to(device)
        converter_state = torch.load(f"{model_path}/embedding_converter.pt", map_location=device)
        converter.load_state_dict(converter_state['state_dict'])
        converter.eval()

        t5_model = t5_model.to(device)
        t5_model.eval()

        print(f"[3/4] Model loaded OK")
        print()

    except Exception as e:
        print(f"ERROR loading model: {str(e)[:100]}")
        return None

    # Evaluate
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    results = []

    print("[4/4] Text restoration")
    print("-"*80)
    print()

    with torch.no_grad():
        for i, (original_text, embedding) in enumerate(zip(test_texts, embeddings)):
            try:
                embedding_tensor = torch.from_numpy(embedding).to(device).unsqueeze(0)
                token_embeddings = converter(embedding_tensor)

                outputs = t5_model.generate(
                    inputs_embeds=token_embeddings,
                    max_length=256,
                    num_beams=1,
                    do_sample=False
                )

                restored_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                original_for_eval = original_text[:100]
                rouge1 = rouge.score(original_for_eval, restored_text)['rouge1'].fmeasure
                rougeL = rouge.score(original_for_eval, restored_text)['rougeL'].fmeasure

                results.append({
                    'idx': i,
                    'rouge1': rouge1,
                    'rougeL': rougeL,
                    'success': len(restored_text) > 0
                })

                status = "OK" if rouge1 > 0 else "NO"
                print(f"Sample {i}: ROUGE={rouge1:.4f} [{status}]")

            except Exception as e:
                results.append({
                    'idx': i,
                    'rouge1': 0,
                    'rougeL': 0,
                    'success': False
                })

    print()
    print("="*80)
    print("[Results]")
    print("="*80)
    print()

    success_count = sum(1 for r in results if r['success'])
    valid_results = [r for r in results if r['success']]

    print(f"Success: {success_count}/{len(results)}")

    if valid_results:
        avg_rouge1 = np.mean([r['rouge1'] for r in valid_results])
        avg_rougeL = np.mean([r['rougeL'] for r in valid_results])
        max_rouge1 = max([r['rouge1'] for r in valid_results])

        print(f"ROUGE-1: avg={avg_rouge1:.4f}, max={max_rouge1:.4f}")
        print(f"ROUGE-L: avg={avg_rougeL:.4f}")
        print()

        if avg_rouge1 > 0.3:
            print("[EXCELLENT] Model is ready for production")
        elif avg_rouge1 > 0.15:
            print("[GOOD] Acceptable quality, ready to deploy")
        elif avg_rouge1 > 0.05:
            print("[FAIR] Partial success, can be used with caution")
        else:
            print("[WEAK] Still needs improvement")
    else:
        print("No successful samples - model needs rework")

    print()
    return results

if __name__ == "__main__":
    # Test large-scale model
    model_path = "./vec_to_text_kor/vec2text/saves/ko_vec2text_1536_v3_large"

    if os.path.exists(model_path):
        results = test_model(model_path, num_samples=20)

        if results:
            # Save results
            with open('final_validation_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to final_validation_results.json")
    else:
        print(f"Model not found: {model_path}")
        print("Run train_large_scale_korean.py first")

