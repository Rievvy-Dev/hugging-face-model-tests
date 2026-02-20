#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mostra as configurações originais dos modelos do HuggingFace."""
from transformers import AutoConfig
from finetuning.config import MODELS

print("\n" + "="*80)
print("CONFIGURAÇÕES ORIGINAIS DOS MODELOS")
print("="*80 + "\n")

for name, model_id in MODELS.items():
    print(f"\n{'='*80}")
    print(f"🤖 {name.upper()}: {model_id}")
    print(f"{'='*80}\n")
    
    try:
        config = AutoConfig.from_pretrained(model_id)
        
        # Informações principais
        print(f"📋 Arquitetura: {config.model_type}")
        if hasattr(config, 'architectures'):
            print(f"   Classe: {config.architectures}")
        
        print(f"\n🔢 Dimensões:")
        if hasattr(config, 'd_model'):
            print(f"   - d_model: {config.d_model}")
        if hasattr(config, 'hidden_size'):
            print(f"   - hidden_size: {config.hidden_size}")
        if hasattr(config, 'num_hidden_layers'):
            print(f"   - num_hidden_layers: {config.num_hidden_layers}")
        if hasattr(config, 'encoder_layers'):
            print(f"   - encoder_layers: {config.encoder_layers}")
        if hasattr(config, 'decoder_layers'):
            print(f"   - decoder_layers: {config.decoder_layers}")
        if hasattr(config, 'num_attention_heads'):
            print(f"   - num_attention_heads: {config.num_attention_heads}")
        if hasattr(config, 'encoder_attention_heads'):
            print(f"   - encoder_attention_heads: {config.encoder_attention_heads}")
        if hasattr(config, 'decoder_attention_heads'):
            print(f"   - decoder_attention_heads: {config.decoder_attention_heads}")
        
        print(f"\n📏 Comprimentos:")
        if hasattr(config, 'max_position_embeddings'):
            print(f"   - max_position_embeddings: {config.max_position_embeddings}")
        if hasattr(config, 'max_length'):
            print(f"   - max_length: {config.max_length}")
        
        print(f"\n🎛️  Vocabulário:")
        if hasattr(config, 'vocab_size'):
            print(f"   - vocab_size: {config.vocab_size}")
        
        print(f"\n⚙️  Ativação & Dropout:")
        if hasattr(config, 'activation_function'):
            print(f"   - activation_function: {config.activation_function}")
        if hasattr(config, 'dropout'):
            print(f"   - dropout: {config.dropout}")
        if hasattr(config, 'attention_dropout'):
            print(f"   - attention_dropout: {config.attention_dropout}")
        if hasattr(config, 'activation_dropout'):
            print(f"   - activation_dropout: {config.activation_dropout}")
        
        print(f"\n🔧 Outras configs:")
        if hasattr(config, 'decoder_start_token_id'):
            print(f"   - decoder_start_token_id: {config.decoder_start_token_id}")
        if hasattr(config, 'eos_token_id'):
            print(f"   - eos_token_id: {config.eos_token_id}")
        if hasattr(config, 'pad_token_id'):
            print(f"   - pad_token_id: {config.pad_token_id}")
        if hasattr(config, 'forced_eos_token_id'):
            print(f"   - forced_eos_token_id: {config.forced_eos_token_id}")
        
        print(f"\n📊 Total de parâmetros estimados:")
        if hasattr(config, 'd_model') and hasattr(config, 'encoder_layers'):
            # Estimativa aproximada para Transformer
            params = config.vocab_size * config.d_model  # Embeddings
            params += config.encoder_layers * (config.d_model ** 2) * 12  # Encoder
            params += config.decoder_layers * (config.d_model ** 2) * 12  # Decoder
            print(f"   ~{params:,} parâmetros")
        
        print(f"\n📄 Config completo disponível em:")
        print(f"   https://huggingface.co/{model_id}/blob/main/config.json")
        
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")

print(f"\n{'='*80}\n")
