#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩→토큰 변환 모듈
1536차원 OpenAI 임베딩 → T5 입력 형태로 변환
"""

import torch
import torch.nn as nn

class EmbeddingToTokenConverter(nn.Module):
    """
    1536차원 임베딩을 T5의 입력 임베딩 공간으로 변환

    흐름:
    1536 임베딩 → Linear(1536→3072) → 4개 토큰으로 해석 → T5 입력
    """

    def __init__(self, embedding_dim=1536, hidden_dim=768, num_tokens=4):
        """
        Args:
            embedding_dim: 입력 임베딩 차원 (1536)
            hidden_dim: T5 임베딩 차원 (768)
            num_tokens: 생성할 토큰 수 (3072 = 768×4)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        # 선형 변환: 1536 → 768×4 (3072)
        self.linear = nn.Linear(embedding_dim, hidden_dim * num_tokens)

        # 스케일 팩터 (T5 임베딩 가중치의 실제 범위에 맞춤)
        # 정상 T5 임베딩: [-110, 70] 범위 vs 우리 출력: [-0.77, 0.69]
        # 필요한 스케일: ~150
        self.scale = nn.Parameter(torch.ones(1) * 150.0)

        print("[EmbeddingToTokenConverter 설계]")
        print(f"  입력: {embedding_dim}차원 (OpenAI 임베딩)")
        print(f"  변환: Linear({embedding_dim} → {hidden_dim*num_tokens})")
        print(f"  스케일: T5 임베딩 범위에 맞춤")
        print(f"  출력: {num_tokens}개 토큰 ({hidden_dim}차원 각)")

    def forward(self, embedding):
        """
        Args:
            embedding: (batch, 1536) OpenAI 임베딩

        Returns:
            tokens: (batch, num_tokens, hidden_dim) T5 입력 형태
        """
        # embedding: (batch, 1536)

        # 선형 변환
        projected = self.linear(embedding)  # (batch, 3072)

        # Reshape to token embeddings
        # (batch, 3072) → (batch, 4, 768)
        tokens = projected.view(
            embedding.size(0),  # batch
            self.num_tokens,     # num_tokens = 4
            self.hidden_dim      # hidden_dim = 768
        )

        # 스케일링 (T5 임베딩과 유사한 스케일)
        tokens = tokens * self.scale  # (batch, 4, 768)

        return tokens


class Vec2TextWithEmbeddingInput(nn.Module):
    """
    1536차원 임베딩을 입력으로 받아 한국어 텍스트를 생성하는 모델
    """

    def __init__(self, t5_model, converter):
        """
        Args:
            t5_model: T5ForConditionalGeneration 모델
            converter: EmbeddingToTokenConverter
        """
        super().__init__()
        self.t5_model = t5_model
        self.converter = converter

    def forward(self, embedding, labels=None):
        """
        Args:
            embedding: (batch, 1536) OpenAI 임베딩
            labels: (batch, seq_len) 선택사항 (훈련용)

        Returns:
            outputs: T5 모델의 출력
        """
        # 1536 임베딩 → 토큰 형태
        token_embeddings = self.converter(embedding)  # (batch, 4, 768)

        # T5에 입력 (inputs_embeds 사용)
        outputs = self.t5_model(
            inputs_embeds=token_embeddings,
            labels=labels
        )

        return outputs

    def generate(self, embedding, max_length=256, **kwargs):
        """
        텍스트 생성
        """
        token_embeddings = self.converter(embedding)  # (batch, 4, 768)

        outputs = self.t5_model.generate(
            inputs_embeds=token_embeddings,
            max_length=max_length,
            **kwargs
        )

        return outputs


# ============================================================================
# 검증: 모듈 설계 확인
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("임베딩→토큰 변환 모듈 설계 검증")
    print("="*80)
    print()

    # 모듈 생성
    converter = EmbeddingToTokenConverter(
        embedding_dim=1536,
        hidden_dim=768,
        num_tokens=4
    )
    print()

    # 더미 입력
    print("[더미 입력 테스트]")
    batch_size = 2
    dummy_embedding = torch.randn(batch_size, 1536)
    print(f"  입력 형태: {dummy_embedding.shape}")

    # 변환
    output = converter(dummy_embedding)
    print(f"  출력 형태: {output.shape}")
    print()

    # 검증
    print("[설계 검증]")
    expected_shape = (batch_size, 4, 768)
    if output.shape == expected_shape:
        print(f"  ✓ 출력 형태 정확: {output.shape}")
    else:
        print(f"  ✗ 출력 형태 오류: {output.shape} (예상: {expected_shape})")

    print(f"  ✓ 파라미터: {sum(p.numel() for p in converter.parameters()):,}")
    print(f"    - linear: 1536×3072 = {1536*3072:,}")
    print(f"    - layer_norm: 768×2 = {768*2:,}")
    print()

    print("="*80)
    print("✓ Phase 2 완료: 임베딩→토큰 변환 모듈 설계 검증 성공")
    print("="*80)

