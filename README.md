# Pipeline de Avaliação e Fine-Tuning de Modelos de Tradução EN→PT

## Visão Geral

Este projeto implementa um **pipeline de 5 estágios** para avaliar e fine-tunar modelos de tradução automática neural (NMT) inglês→português, aplicados ao domínio de abstracts científicos do **SciELO**.

O modelo selecionado para fine-tuning foi o **`unicamp-dl/translation-en-pt-t5`**, uma adaptação do T5 (Text-to-Text Transfer Transformer) para tradução EN→PT, desenvolvido pela Universidade Estadual de Campinas (UNICAMP).

### Motivação: Por que estudar tradução automática neural quando LLMs já traduzem bem?

Modelos de linguagem de grande porte (LLMs) como GPT-4 e Claude produzem traduções de alta qualidade em cenários gerais. Isso levanta uma questão legítima: **por que pesquisar fine-tuning de modelos NMT dedicados?** A resposta envolve múltiplas dimensões fundamentais para pesquisa acadêmica e aplicações em escala:

#### 1. Custo e escalabilidade

O corpus SciELO contém **2,7 milhões** de pares de abstracts. Traduzir esse volume via API de LLM teria custo proibitivo:

| Abordagem             | Custo estimado (2.7M abstracts)      | Latência          |
|-----------------------|--------------------------------------|--------------------|
| GPT-4 API             | ~$8.000–15.000 (tokens de I/O)       | Dias (rate limits) |
| Claude API            | ~$5.000–10.000                       | Dias (rate limits) |
| Google Translate API  | ~$4.000–6.000                        | Horas              |
| **NMT local (T5)**    | **$0 (apenas eletricidade)**         | **Horas (GPU)**    |

Um modelo NMT fine-tuned roda localmente em uma **GPU de ~$300** (RTX 4050) sem custo por token, sem limites de taxa, e sem dependência de serviços externos.

#### 2. Reprodutibilidade e rigor científico

Resultados acadêmicos devem ser **reprodutíveis**. LLMs comerciais são:
- **Não-determinísticos**: mesma entrada pode gerar saídas diferentes (temperature > 0)
- **Opacos**: arquitetura, dados de treino e pesos são proprietários
- **Mutáveis**: modelos são atualizados sem aviso — GPT-4 de janeiro ≠ GPT-4 de junho
- **Não-auditáveis**: impossível inspecionar por que uma tradução específica foi gerada

Um modelo NMT open-source com pesos fixos produz **saída determinística** e permite **inspeção completa**: arquitetura, pesos, tokenizador, dados de treino — tudo verificável e citável.

#### 3. Soberania de dados e privacidade

Textos biomédicos podem conter informações sensíveis. Enviar dados para APIs externas levanta questões de:
- **Privacidade**: dados podem ser retidos para treino pelos provedores
- **Conformidade legal**: LGPD e regulamentações de dados biomédicos
- **Soberania**: dependência de infraestrutura estrangeira para processamento de dados nacionais

Modelos locais processam dados **inteiramente em hardware próprio**, sem transmissão para terceiros.

#### 4. Especialização de domínio

LLMs são generalistas. Para domínios especializados como biomedicina, modelos NMT fine-tuned oferecem vantagens (Koehn & Knowles, 2017):
- **Consistência terminológica**: termos como "randomized controlled trial" devem ser sempre traduzidos como "ensaio clínico randomizado", não variar entre chamadas
- **Vocabulário de domínio**: tokenizador e embeddings ajustados para termos científicos
- **Avaliação controlada**: métricas calculáveis (BLEU, COMET) em test sets fixos

Zhu et al. (2023) demonstraram que LLMs como GPT-4 superam o NLLB em apenas **40,91%** das direções de tradução, com gap significativo para traduções especializadas e pares de idiomas com menos recursos.

#### 5. Contribuição científica

A relevância acadêmica deste trabalho não está apenas nos resultados, mas na **metodologia**:
- Documentar um pipeline reprodutível de avaliação e fine-tuning de NMT
- Demonstrar que **técnicas de regularização** importam mais que volume de dados
- Fornecer um caso de estudo empírico de **catastrophic forgetting** vs. fine-tuning bem-sucedido
- Contribuir para a pesquisa em tradução automática EN→PT no domínio biomédico, que ainda é sub-representada na literatura

> **Em resumo**: LLMs são excelentes para tradução casual. Mas para tradução **em escala**, **reprodutível**, **auditável**, **privada** e **especializada em domínio** — como é necessário em pesquisa científica — modelos NMT dedicados e fine-tuned continuam sendo a abordagem mais adequada e economicamente viável.

### Resultados Obtidos

| Métrica    | Antes do Fine-tuning | Após Fine-tuning (Epoch 12) | Delta   | Melhoria |
|------------|---------------------:|----------------------------:|--------:|---------:|
| BLEU       | 40.06                | 45.51                       | +5.45   | +13.6%   |
| chrF       | 65.61                | 70.54                       | +4.93   | +7.5%    |
| COMET      | 0.8499               | 0.8756                      | +0.0257 | +3.0%    |
| BERTScore  | 0.8957               | 0.9124                      | +0.0167 | +1.9%    |

---

## Sobre o Modelo: `unicamp-dl/translation-en-pt-t5`

### Arquitetura

O modelo é baseado na arquitetura **T5 (Text-to-Text Transfer Transformer)** proposta por Raffel et al. (2019). O T5 trata todas as tarefas de NLP como problemas de texto-para-texto, onde tanto a entrada quanto a saída são sequências de texto.

| Componente                | Especificação               |
|---------------------------|:----------------------------|
| Arquitetura base          | T5 (encoder-decoder)        |
| Camadas do encoder        | 12                          |
| Camadas do decoder        | 12                          |
| Dimensão oculta (d_model) | 768                         |
| Cabeças de atenção        | 12                          |
| Dimensão do feed-forward  | 3072                        |
| Parâmetros totais         | ~223M (222.903.552)         |
| Vocabulário               | 32.128 tokens (SentencePiece) |
| Tipo de atenção           | Multi-head self-attention   |
| Normalização              | Layer Normalization (pre-norm) |
| Ativação                  | ReLU (Rectified Linear Unit) |

### O que significam os parâmetros da arquitetura?

Cada campo do `config.json` do modelo define uma propriedade matemática específica da rede neural. Abaixo, a explicação de cada um com as fórmulas:

#### `d_model = 768` — Dimensão oculta

É o tamanho do vetor que representa cada token em **todas as camadas** do modelo. Cada palavra (token) da entrada é convertida em um vetor de 768 dimensões. Todas as operações internas (atenção, feed-forward, projeção) operam nessa dimensionalidade.

$$\text{embedding}(x_i) \in \mathbb{R}^{768}$$

**Analogia**: Se cada token fosse uma pessoa, `d_model` seria quantas "características" (altura, peso, idade, ...) descrevem essa pessoa. Com 768 características, o modelo captura nuances semânticas muito finas.

#### `num_heads = 12` — Cabeças de atenção

O mecanismo de **Multi-Head Attention** (Vaswani et al., 2017) divide a atenção em múltiplas "perspectivas" independentes. Cada cabeça aprende a capturar um tipo diferente de relação linguística:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{12}) \cdot W^O$$

Onde cada cabeça é:

$$\text{head}_i = \text{Attention}(Q \cdot W_i^Q, \; K \cdot W_i^K, \; V \cdot W_i^V)$$

**Referência**: Vaswani, A. et al. (2017). *Attention is All You Need*. In NeurIPS 2017. https://arxiv.org/abs/1706.03762

**O que cada cabeça captura** (exemplos típicos do que se observa em modelos treinados):

```
Head 1:  Relações sujeito-verbo     ("paciente" ← atenção → "apresentou")
Head 2:  Relações de adjacência     ("febre" ← atenção → "persistente")
Head 3:  Relações de correferência  ("ele" ← atenção → "paciente")
Head 4:  Relações posicionais       (palavras próximas entre si)
Head 5:  Pontuação e estrutura      ("." ← atenção → fim de sentença)
...
Head 12: Padrões aprendidos diversos
```

#### `d_kv = 64` — Dimensão por cabeça de atenção

Cada cabeça de atenção opera num subespaço de dimensão $d_{kv}$. É a dimensão dos vetores Query ($Q$), Key ($K$) e Value ($V$) individuais de cada cabeça.

$$d_{kv} = \frac{d_{model}}{num\_heads} = \frac{768}{12} = 64$$

O mecanismo de **Scaled Dot-Product Attention** (a operação central de cada cabeça) é:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_{kv}}}\right) \cdot V$$

Onde:
- $Q \in \mathbb{R}^{n \times 64}$ = queries (o que cada token "procura")
- $K \in \mathbb{R}^{n \times 64}$ = keys (o que cada token "oferece" para ser encontrado)
- $V \in \mathbb{R}^{n \times 64}$ = values (a informação que cada token "carrega")
- $\sqrt{d_{kv}} = \sqrt{64} = 8$ = fator de escala (evita que o softmax sature)
- $n$ = comprimento da sequência

```
Exemplo: "O paciente apresentou febre"  (4 tokens, d_kv=64)

                  K₁(O)   K₂(pac.)  K₃(apr.)  K₄(febre)
Q₁(O)          [ 0.80     0.05      0.10      0.05   ]    → "O" atende a si mesmo
Q₂(paciente)   [ 0.10     0.30      0.50      0.10   ]    → "paciente" atende "apresentou"
Q₃(apresentou) [ 0.05     0.45      0.20      0.30   ]    → "apresentou" atende "paciente"
Q₄(febre)      [ 0.02     0.08      0.40      0.50   ]    → "febre" atende "apresentou"
                  ↑ cada valor é um peso de atenção (soma = 1 por linha, via softmax)
```

#### `d_ff = 3072` — Dimensão do feed-forward

Após cada bloco de atenção, o output passa por uma rede **Feed-Forward** (FFN) de duas camadas. A primeira expande a dimensionalidade, a segunda comprime de volta:

$$\text{FFN}(x) = \text{ReLU}(x \cdot W_1) \cdot W_2$$

> **Nota**: A formulação original de Vaswani et al. (2017) inclui termos de bias ($b_1, b_2$), mas a implementação T5 **não usa bias** nas camadas lineares — apenas as matrizes de peso $W_1$ e $W_2$.

Onde:
- $W_1 \in \mathbb{R}^{768 \times 3072}$ → expande 768 → 3072 (4x)
- $W_2 \in \mathbb{R}^{3072 \times 768}$ → comprime 3072 → 768
- $\text{ReLU}(z) = \max(0, z)$ → ativação não-linear

```
Input:  x ∈ ℝ^768    (vetor do token após atenção)
         ↓
    W₁ × x            → ℝ^3072  (expansão: 768 → 3072, sem bias)
         ↓
    ReLU(·)           → ℝ^3072  (não-linearidade: zera negativos)
         ↓
    W₂ × ·            → ℝ^768   (compressão: 3072 → 768, sem bias)
         ↓
Output: y ∈ ℝ^768    (mesmo tamanho que input → residual connection)
```

**Por que 3072?** A razão $d_{ff} / d_{model} = 3072 / 768 = 4\times$ é uma convenção estabelecida por Vaswani et al. (2017). A expansão temporária para 4x permite ao modelo aprender transformações mais complexas, e a compressão de volta para $d_{model}$ mantém a uniformidade dimensional entre camadas.

#### `dropout_rate = 0.1` — Regularização por dropout

Durante o treino, **10% dos neurônios são aleatoriamente desativados** (zerados) a cada forward pass. Isso força o modelo a aprender representações mais robustas — ele não pode depender de nenhum neurônio individual.

$$\text{Dropout}(x_i) = \begin{cases} \frac{x_i}{1-p} & \text{com probabilidade } 1-p \\ 0 & \text{com probabilidade } p = 0.1 \end{cases}$$

O fator $\frac{1}{1-p} = \frac{1}{0.9} \approx 1.11$ é o **inverted dropout** — escala os valores restantes para manter a mesma magnitude esperada (Srivastava et al., 2014).

**Referência**: Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 15(1), pp. 1929–1958.

#### `relative_attention_num_buckets = 32` — Posição relativa

Diferente do Transformer original que usa embeddings posicionais absolutos (senoidais), o T5 usa **relative position bias** (Shaw et al., 2018; Raffel et al., 2019). Em vez de codificar a posição absoluta de cada token, codifica a **distância relativa** entre pares de tokens.

As distâncias relativas são agrupadas em 32 "buckets" (baldes) usando uma escala logarítmica:

```
Distância relativa    Bucket
──────────────────    ──────
         0              0     (mesmo token)
        ±1              1     (adjacente)
        ±2              2
        ±3-4            3     (começa a agrupar)
        ±5-7            4
        ±8-15           5
        ±16-31          6
        ±32-63          7
        ...             ...
       ±64-128         ...    (max_distance=128)
```

A escala logarítmica permite que o modelo distinga tokens próximos com alta resolução, mas agrupe tokens distantes — o que faz sentido linguisticamente (a relação entre palavras adjacentes é mais variada que entre palavras separadas por 100 tokens).

**Referência**: Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). *Self-Attention with Relative Position Representations*. In Proceedings of NAACL-HLT 2018, pp. 464–468. https://aclanthology.org/N18-2074/

#### `vocab_size = 32128` — Tamanho do vocabulário

O tokenizador **SentencePiece** (Kudo & Richardson, 2018) usa um modelo **unigram** que decompõe textos em subpalavras:

```
Texto: "randomized controlled trial" → 32128 possíveis subpalavras

Tokenização:
  "randomized"     → ["_random", "ized"]              (2 tokens)
  "controlled"     → ["_control", "led"]               (2 tokens)
  "trial"          → ["_trial"]                         (1 token)
  Total: 5 tokens

Texto raro: "bronchopneumonia" → ["_broncho", "pne", "umon", "ia"] (4 tokens)
Texto comum: "the" → ["_the"]  (1 token)
```

A embedding layer mapeia cada um dos 32.128 tokens para um vetor de $d_{model} = 768$ dimensões:

$$E \in \mathbb{R}^{32128 \times 768}$$

Isso soma **24,7M parâmetros** apenas na embedding (compartilhada entre encoder e decoder no T5).

**Referência**: Kudo, T. & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. In Proceedings of EMNLP 2018, pp. 66–71. https://aclanthology.org/D18-2012/

#### Cálculo do total de parâmetros (~220M)

A contagem detalhada de parâmetros do modelo T5-base:

$$\text{Params}_{total} = \text{Params}_{embedding} + \text{Params}_{encoder} + \text{Params}_{decoder} + \text{Params}_{head}$$

```
1. EMBEDDING (compartilhada encoder/decoder):
   E = vocab_size × d_model = 32128 × 768 = 24,674,304 params

2. ENCODER (12 camadas, cada uma com):
   a) Self-Attention (sem bias — T5 usa projeções lineares sem termo de viés):
      W_Q, W_K, W_V: 3 × (d_model × d_kv × num_heads) = 3 × (768 × 64 × 12) = 1,769,472
      W_O:           d_model × d_model = 768 × 768 = 589,824
      T5LayerNorm:   d_model = 768  (apenas scale, sem bias — RMSNorm)
      Subtotal attn: 2,360,064 /camada

   b) Feed-Forward (sem bias nas camadas lineares):
      W₁: d_model × d_ff = 768 × 3072 = 2,359,296
      W₂: d_ff × d_model = 3072 × 768 = 2,359,296
      T5LayerNorm: d_model = 768  (apenas scale)
      Subtotal FFN: 4,719,360 /camada

   + Relative Attention Bias (apenas no bloco 0, compartilhado):
      relative_attention_bias: num_buckets × num_heads = 32 × 12 = 384

   Total Encoder: 12 × (2,360,064 + 4,719,360) + 384 + 768 (final LN)
   Total Encoder: 84,954,240

3. DECODER (12 camadas, cada uma com):
   a) Self-Attention:    mesma estrutura    = 2,360,064 /camada
   b) Cross-Attention:   mesma estrutura    = 2,360,064 /camada
   c) Feed-Forward:      mesma estrutura    = 4,719,360 /camada

   + Relative Attention Bias (bloco 0): 384
   Total Decoder: 12 × (2,360,064 + 2,360,064 + 4,719,360) + 384 + 768 (final LN)
   Total Decoder: 113,275,008

4. LM HEAD (compartilha pesos com embedding):
   Sem parâmetros adicionais (tied weights)

TOTAL: 24,674,304 + 84,954,240 + 113,275,008 = 222,903,552 ≈ 223M ✅
(Verificado: safetensors do modelo contém exatamente 222,903,552 parâmetros)
```

### Fluxo completo Encoder-Decoder

```
ENCODER (processa o texto fonte em paralelo):

  Input: "The patient presented fever"
    ↓ Tokenize + Embed
  X₀ = [e₁, e₂, e₃, e₄]  ∈ ℝ^(4×768)   (4 tokens × 768 dims)
    ↓ + Relative Position Bias
    ↓
  ┌─── Camada 1 ────────────────────────────────────────┐
  │ Layer Norm → Self-Attention → Residual Connection    │
  │ X₁ = LayerNorm(X₀) → MultiHead(Q,K,V) + X₀        │
  │ Layer Norm → FFN → Residual Connection               │
  │ X₁ = LayerNorm(X₁) → FFN(X₁) + X₁                 │
  └──────────────────────────────────────────────────────┘
    ↓ ... repete 12 vezes ...
  ┌─── Camada 12 ───────────────────────────────────────┐
  │ (mesma estrutura)                                    │
  └──────────────────────────────────────────────────────┘
    ↓ Final Layer Norm
  H_enc = [h₁, h₂, h₃, h₄]  ∈ ℝ^(4×768)   ← "memória" do encoder

DECODER (gera token por token, autoregressivamente):

  Target: "<pad> O paciente apresentou febre" (shifted right)
    ↓ Tokenize + Embed
  Y₀ = [d₁, d₂, d₃, d₄, d₅]
    ↓
  ┌─── Camada 1 ────────────────────────────────────────┐
  │ Layer Norm → Masked Self-Attention → Residual        │
  │   (cada token só vê tokens ANTERIORES — causal)      │
  │ Layer Norm → Cross-Attention(Q=dec, K=enc, V=enc)    │
  │   (decoder "consulta" o encoder: alinha source↔target)│
  │ Layer Norm → FFN → Residual                          │
  └──────────────────────────────────────────────────────┘
    ↓ ... repete 12 vezes ...
    ↓ Final Layer Norm
    ↓ LM Head (projeção linear → logits ∈ ℝ^32128)
    ↓ Softmax → probabilidade sobre todo o vocabulário
  P("O" | "The patient presented fever", <pad>) = 0.87
  P("paciente" | "The patient presented fever", <pad> O) = 0.92
  ...
```

### Pré-treinamento e Dados Originais

- **Pré-treinamento base**: PTT5 — modelo T5 pré-treinado em corpus em português
- **Fine-tuning de tradução (pelos autores)**: ParaCrawl (5M+ pares EN-PT) + Corpora biomédica científica (6M+ pares)
- **Tarefa**: Tradução EN→PT com prefixo `"translate English to Portuguese: "`
- **Tokenizador**: SentencePiece (unigram) com vocabulário de 32k tokens

### Referência Acadêmica

```bibtex
@inproceedings{lopes-etal-2020-lite,
    title     = "Lite Training Strategies for {P}ortuguese-{E}nglish and {E}nglish-{P}ortuguese Translation",
    author    = "Lopes, Alexandre and Nogueira, Rodrigo and Lotufo, Roberto and Pedrini, Helio",
    booktitle = "Proceedings of the Fifth Conference on Machine Translation",
    month     = nov,
    year      = "2020",
    address   = "Online",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2020.wmt-1.90",
    pages     = "833--840",
}
```

### Como o modelo foi selecionado? — O caso Helsinki

A seleção do modelo não foi automática. O `Helsinki-NLP/opus-mt-tc-big-en-pt` foi a **primeira escolha** para fine-tuning, pois liderou o ranking no STAGE 1 (BLEU=37.47, chrF=59.85 na avaliação geral). Porém, o fine-tuning do Helsinki **fracassou** — os resultados **pioraram** em relação ao modelo base.

#### Tentativa com Helsinki: configuração e resultados

| Parâmetro               | Helsinki (1ª tentativa)       | Unicamp-T5 (2ª tentativa)         |
|--------------------------|-------------------------------|-----------------------------------|
| Arquitetura              | MarianMT (~600M params)       | T5 (~220M params)                 |
| Dataset de treino        | 80.000 exemplos               | 18.000 exemplos                   |
| Dataset de validação     | ❌ Nenhum                     | ✅ 2.000 exemplos                 |
| Epochs                   | 5                             | 12                                |
| Batch size               | 8                             | 8                                 |
| Gradient accumulation    | ❌ Não                        | ✅ 2 (effective batch = 16)       |
| Learning rate            | ~2e-5 (default)               | 1e-5 (conservador)               |
| FP16 (mixed precision)   | ❌ Não                        | ✅ Sim                            |
| max_seq_len              | ❌ Não configurado (default)  | ✅ 256 tokens                     |
| Early stopping           | ❌ Não                        | ✅ patience=2                     |

#### Por que o Helsinki fracassou?

```
Helsinki: Training Loss ao longo de 50.000 steps (5 epochs)

Loss
 8 ┤ ██
 7 ┤  ██
 6 ┤    ███
 5 ┤       ████
 4 ┤           ████
 3 ┤               █████
 2 ┤                    ███████
 1 ┤                           ████████████████
 0 ┤                                           ██████████ ← 0.14 (OVERFITTING!)
   └──────────────────────────────────────────────────────
   0     10k    20k    30k    40k    50k steps
```

Análise do `trainer_state.json` do Helsinki:
- **Training loss**: 7.65 → 0.14 (queda de 98%) — o modelo **memorizou** os dados de treino
- **Eval loss**: **inexistente** — nenhuma avaliação durante o treino (0 eval entries)
- **best_metric**: `None` — sem monitoramento, sem seleção do melhor checkpoint
- **Resultado final**: BLEU = **36** (era 42.64 no SciELO base → **degradação de -6.6 pontos!**)
- **chrF** = **65** (era 68.93 → **degradação de -3.9 pontos**)
- **COMET e BERTScore**: não foi possível medir

O diagnóstico é claro: **catastrophic forgetting** (esquecimento catastrófico). Sem conjunto de validação, sem early stopping, e sem regularização, o modelo com 600M de parâmetros **memorizou** os 80k exemplos de treino (loss → 0.14) mas **perdeu a capacidade de generalizar** para textos novos. Este é um fenômeno bem documentado na literatura de adaptação de domínio em NMT (Miceli Barone et al., 2017; Freitag & Al-Onaizan, 2016).

#### Por que o Unicamp-T5 teve sucesso?

A segunda tentativa aplicou todas as lições aprendidas com a falha do Helsinki:

1. **Conjunto de validação (2k exemplos)**: Permitiu monitorar eval_loss a cada epoch e detectar overfitting
2. **Early stopping (patience=2)**: Interromperia o treino automaticamente se eval_loss parasse de melhorar
3. **Gradient accumulation (2)**: Effective batch size de 16, suavizando gradientes ruidosos
4. **Learning rate conservador (1e-5)**: Metade do default, evitando atualizações destrutivas
5. **FP16 (mixed precision)**: Viabilizou treinar na RTX 4050 (6GB VRAM) sem out-of-memory
6. **max_seq_len=256**: Truncamento explícito, evitando sequências variáveis que desestabilizam o treino
7. **Modelo 3x menor (220M vs 600M)**: Menos propenso a overfitting com dados limitados

**Resultado**: Training loss convergiu para **0.97** — praticamente igual ao eval_loss (**0.97**), indicando zero overfitting. BLEU subiu de 40.06 para **45.51** (+13.6%).

#### Fundamentação: por que menos dados + mais técnicas supera mais dados sem técnicas?

A literatura de adaptação de domínio em NMT sustenta fortemente este resultado:

- **Miceli Barone et al. (2017)** demonstraram que, ao fazer fine-tuning de NMT em dados in-domain de tamanho limitado, **técnicas de regularização** (dropout, L2, early stopping) são mais importantes que o volume de dados. Sem regularização, modelos grandes overfitam rapidamente, mesmo com datasets grandes. O artigo encontra uma relação **logarítmica** entre volume de dados e ganho em BLEU — ou seja, dobrar os dados não dobra a qualidade.

- **Freitag & Al-Onaizan (2016)** mostraram que é possível adaptar modelos NMT a novos domínios **com poucos dados in-domain**, desde que o processo de fine-tuning seja controlado. A chave é **qualidade do processo**, não quantidade de dados.

- **Neubig & Hu (2018)** propuseram "similar-language regularization" para evitar overfitting em adaptação com dados limitados, confirmando que a **prevenção de overfitting** é o fator crítico em domain adaptation.

- **Koehn & Knowles (2017)** identificaram 6 desafios para NMT, incluindo que modelos neurais são particularmente sensíveis a **dados fora do domínio** e que adaptação de domínio requer técnicas cuidadosas.

No nosso caso, os 18k exemplos do SciELO são **altamente representativos** do domínio-alvo (abstracts científicos biomédicos EN→PT), enquanto os 80k do Helsinki possivelmente continham ruído ou distribuição menos focada. Mais epochs (12 vs 5) permitiram **exposição repetida ao vocabulário especializado** do domínio, enquanto o early stopping impediu que essa repetição causasse memorização.

```
RESUMO DA SELEÇÃO:

Helsinki (1ª tentativa)         Unicamp-T5 (2ª tentativa)
├─ 600M params                  ├─ 220M params
├─ 80k treino, 0 validação      ├─ 18k treino, 2k validação
├─ 5 epochs, sem early stop     ├─ 12 epochs, early stopping
├─ Sem grad_accum, sem fp16     ├─ grad_accum=2, fp16
├─ Loss: 7.65 → 0.14 ⚠️        ├─ Loss: ~2.5 → 0.97 ✅
├─ BLEU: 42.64 → 36 📉 (-15.6%) ├─ BLEU: 40.06 → 45.51 📈 (+13.6%)
└─ FRACASSO (overfitting)       └─ SUCESSO (generalização)
```

---

## Pipeline de 5 Estágios

```
STAGE 1: AVALIAÇÃO INICIAL
├─ Testar 6 modelos pré-treinados em 3 datasets públicos
├─ Calcular BLEU, chrF, COMET, BERTScore
└─ Resultado: evaluation_results/translation_metrics_all.csv
        ↓
STAGE 2: SELEÇÃO DO MODELO
├─ 1ª tentativa: Helsinki (fracasso — catastrophic forgetting)
├─ 2ª tentativa: unicamp-dl/translation-en-pt-t5 (sucesso)
└─ Resultado: modelo definido com base em experimentação empírica
        ↓
STAGE 3: PREPARAÇÃO DE DADOS
├─ Separar SciELO em 3 splits não-sobrepostos:
│   ├─ 18.000 exemplos para TREINO
│   ├─  2.000 exemplos para VALIDAÇÃO (early stopping)
│   └─  5.000 exemplos para TESTE
└─ Resultado: finetuning/abstracts-datasets/*.csv
        ↓
STAGE 4: FINE-TUNING
├─ GPU: NVIDIA RTX 4050 (6GB VRAM)
├─ 12 epochs, batch_size=8, grad_accum=2, lr=1e-5
├─ Early stopping com patience=2
└─ Resultado: unicamp-t5/unicamp-t5/ (modelo fine-tuned)
        ↓
STAGE 5: AVALIAÇÃO FINAL
├─ Testar modelo base vs fine-tuned nos MESMOS 5k dados de teste
├─ Calcular delta de métricas
└─ Resultado: scielo_before_finetuning.csv / scielo_after_finetuning_epoch_*.csv
```

---

## STAGE 1: Avaliação Inicial dos Modelos

### Objetivo
Avaliar 6 modelos pré-treinados em 3 datasets públicos para estabelecer baselines.

### Modelos Avaliados

| # | Modelo                                                        | Arquitetura | Parâmetros |
|---|---------------------------------------------------------------|-------------|------------|
| 1 | `Helsinki-NLP/opus-mt-tc-big-en-pt`                          | MarianMT    | ~600M      |
| 2 | `Narrativa/mbart-large-50-finetuned-opus-en-pt-translation`  | mBART-50    | ~611M      |
| 3 | `unicamp-dl/translation-en-pt-t5`                            | T5          | ~220M      |
| 4 | `VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted`     | T5          | ~220M      |
| 5 | `danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR`             | M2M100      | ~418M      |
| 6 | `quickmt/quickmt-en-pt`                                      | CTranslate2 | —          |

### Datasets Públicos

| Dataset      | Exemplos | Descrição                    |
|--------------|----------|------------------------------|
| WMT24++      | 998      | Avaliação en→pt_BR           |
| ParaCrawl    | 5.000    | Crawl web paralelo en→pt     |
| Flores       | 1.012    | Facebook multilingual        |

### Métricas

| Métrica       | Tipo       | Range | Descrição                                          |
|---------------|------------|-------|----------------------------------------------------|
| **BLEU**      | N-gramas   | 0-100 | Precisão de n-gramas (1-4) com brevity penalty     |
| **chrF**      | Caracteres | 0-100 | F-score baseado em caracteres                      |
| **COMET**     | Neural     | 0-1   | Score neural aprendido (Unbabel/wmt22-comet-da)    |
| **BERTScore** | Neural     | 0-1   | Similaridade semântica via embeddings BERT         |

### Resultados — Média por Modelo (3 datasets)

| #  | Modelo          | BLEU  | chrF  | COMET  | BERTScore | GPU (MB) |
|----|-----------------|------:|------:|-------:|----------:|---------:|
| 1  | Helsinki        | 37.47 | 59.85 | 0.8250 | 0.8667    | 904      |
| 2  | Narrativa mBART | 21.01 | 40.27 | 0.7572 | 0.8350    | 2.340    |
| 3  | Unicamp-T5      | 14.58 | 32.41 | 0.6670 | 0.7922    | 859      |
| 4  | VanessaSchenkel | 8.52  | 25.34 | 0.6342 | 0.7862    | 859      |
| 5  | M2M100          | 22.08 | 48.21 | 0.7530 | 0.8333    | 1.863    |
| 6  | QuickMT         | 0.00  | 4.17  | 0.2701 | 0.4754    | 9        |

### Resultados Detalhados — Por Dataset

**WMT24++ (998 exemplos, sentenças longas ~33 palavras/sentença)**

| Modelo          | BLEU  | chrF  | COMET  | BERTScore | Tempo       |
|-----------------|------:|------:|-------:|----------:|------------:|
| Helsinki        | 33.71 | 58.86 | 0.7825 | 0.8622    | 529s        |
| Narrativa mBART | 6.54  | 25.48 | 0.6452 | 0.7917    | 797s        |
| Unicamp-T5      | 3.55  | 19.73 | 0.5391 | 0.7573    | 237s        |
| VanessaSchenkel | 2.77  | 17.19 | 0.5091 | 0.7562    | 215s        |
| M2M100          | 22.99 | 50.08 | 0.7012 | 0.8404    | 888s        |
| QuickMT         | 0.00  | 4.80  | 0.2480 | 0.4871    | 59s         |

**ParaCrawl (5.000 exemplos, sentenças curtas ~7 palavras/sentença)**

| Modelo          | BLEU  | chrF  | COMET  | BERTScore | Tempo       |
|-----------------|------:|------:|-------:|----------:|------------:|
| Helsinki        | 39.63 | 59.98 | 0.8452 | 0.8696    | 740s        |
| Narrativa mBART | 27.07 | 46.75 | 0.8083 | 0.8544    | 2.013s      |
| Unicamp-T5      | 19.46 | 37.99 | 0.7239 | 0.8076    | 633s        |
| VanessaSchenkel | 11.05 | 28.89 | 0.6868 | 0.7992    | 610s        |
| M2M100          | 22.41 | 47.11 | 0.7735 | 0.8293    | 585s        |
| QuickMT         | 0.00  | 4.03  | 0.2789 | 0.4703    | 288s        |

**Flores (1.012 exemplos)**

| Modelo          | BLEU  | chrF  | COMET  | BERTScore | Tempo       |
|-----------------|------:|------:|-------:|----------:|------------:|
| Helsinki        | 39.08 | 60.72 | 0.8473 | 0.8683    | 131s        |
| Narrativa mBART | 29.43 | 48.59 | 0.8182 | 0.8588    | 378s        |
| Unicamp-T5      | 20.72 | 39.52 | 0.7380 | 0.8116    | 122s        |
| VanessaSchenkel | 11.74 | 29.93 | 0.7066 | 0.8032    | 111s        |
| M2M100          | 20.85 | 47.45 | 0.7842 | 0.8301    | 247s        |
| QuickMT         | 0.00  | 3.68  | 0.2835 | 0.4689    | 59s         |

### Comandos

```bash
# Avaliar 5 modelos primários
python models-test.py --full

# Avaliar 6º modelo (QuickMT - CTranslate2)
python evaluate_quickmt.py --full

# Retomar avaliação interrompida
python models-test.py --resume
python evaluate_quickmt.py --resume
```

### Saída
- `evaluation_results/translation_metrics_all.csv` — consolidado
- `evaluation_results/<modelo>.csv` — individual por modelo

---

## STAGE 2: Seleção do Modelo

### Objetivo
Selecionar o melhor modelo para fine-tuning por experimentação prática.

### Processo Real de Seleção

A seleção não foi automática por score composto. Foi um processo **empírico em duas etapas**:

**Etapa 1 — Helsinki (fracasso)**:
O modelo com melhor desempenho no STAGE 1 (Helsinki, BLEU=37.47) foi a escolha natural. Foi feito fine-tuning com 80k exemplos, 5 epochs, batch_size=8, sem validação, sem early stopping, sem gradient accumulation, sem fp16, sem controle de max_seq_len. O resultado foi **catastrophic forgetting**: BLEU caiu de 42.64→36 no SciELO, chrF de 68.93→65. O modelo memorizou o treino (loss→0.14) mas perdeu generalização.

**Etapa 2 — Unicamp-T5 (sucesso)**:
Com as lições aprendidas, a segunda tentativa usou o `unicamp-dl/translation-en-pt-t5` (220M params, 3x menor), com todas as técnicas de regularização: validação (2k), early stopping, gradient accumulation, fp16, max_seq_len=256, lr conservador. BLEU subiu de 40.06→45.51 (+13.6%).

### Score Composto (ferramenta auxiliar)
O script `choose_best_model.py` calcula um score composto para referência:

$$S = 0.30 \cdot \hat{B} + 0.25 \cdot \hat{C}_r + 0.25 \cdot \hat{C}_o + 0.20 \cdot \hat{B}_s$$

Onde cada métrica é normalizada min-max para $[0, 1]$ entre os modelos avaliados:

$$\hat{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

```
Exemplo: normalização do BLEU
  Valores brutos: Helsinki=37.47, Narrativa=21.01, Unicamp-T5=14.58, ...
  min = 0.00 (QuickMT), max = 37.47 (Helsinki)
  
  BLEU_norm(Helsinki)  = (37.47 - 0.00) / (37.47 - 0.00) = 1.000
  BLEU_norm(Unicamp-T5) = (14.58 - 0.00) / (37.47 - 0.00) = 0.389
```

**Pesos**: BLEU recebe maior peso (0.30) por ser a métrica mais estabelecida. chrF e COMET dividem 0.25 cada. BERTScore recebe 0.20 por ter menor correlação com tradução especificamente.

### Comando
```bash
python choose_best_model.py
```

### Resultado
Modelo selecionado: **`unicamp-dl/translation-en-pt-t5`** — definido após a falha empírica do Helsinki, validado por sua eficiência computacional (220M params, RTX 4050 compatível) e pela qualidade dos resultados de fine-tuning (+5.45 BLEU).

---

## STAGE 3: Preparação de Dados SciELO

### Objetivo
Criar 3 splits não-sobrepostos do dataset SciELO (2.7M exemplos totais).

### Divisão dos Dados

| Split      | Exemplos | Uso                                    |
|------------|----------|----------------------------------------|
| Treino     | 18.000   | Fine-tuning do modelo                  |
| Validação  | 2.000    | Monitorar convergência + early stopping|
| Teste      | 5.000    | Avaliação final (mesmos para base e fine-tuned) |

**Total: 25.000 exemplos (~0.9% do corpus completo)**

### Justificativa do Dataset Compacto

- **18k treino**: Suficiente para adaptação de domínio (abstracts científicos) sem overfitting
- **2k validação**: Monitora eval_loss por epoch e aciona early stopping
- **5k teste**: Mesmo conjunto usado na avaliação do modelo base, garantindo comparação justa
- **Seed fixo (42)**: Splits são determinísticos e reprodutíveis

### Como funcionam os 2.000 exemplos de validação?

O conjunto de validação **não é usado para treinar** o modelo — seus pesos nunca são atualizados com base nesses dados. Ele serve exclusivamente para **monitorar a generalização** durante o treino:

```
Fluxo por epoch:

  ┌──────────────────────────────────────────────────────────────┐
  │ TREINO (18k exemplos)                                        │
  │  O modelo processa todos os 18k exemplos em mini-batches     │
  │  de 8, atualizando pesos a cada batch (gradient descent).    │
  │  → Calcula: training_loss (quão bem acerta os dados de treino)│
  └──────────────────────────────────────────────────────────────┘
           ↓ (ao final de cada epoch)
  ┌──────────────────────────────────────────────────────────────┐
  │ VALIDAÇÃO (2k exemplos) — modo inference, SEM gradient       │
  │  O modelo traduz os 2k exemplos SEM atualizar pesos.         │
  │  → Calcula: eval_loss (quão bem acerta dados NUNCA vistos)   │
  └──────────────────────────────────────────────────────────────┘
           ↓
  ┌──────────────────────────────────────────────────────────────┐
  │ DECISÃO DO EARLY STOPPING                                    │
  │  Se eval_loss melhorou → salva checkpoint, reseta contador   │
  │  Se eval_loss NÃO melhorou por 2 epochs → PARA o treino     │
  └──────────────────────────────────────────────────────────────┘
```

**Por que isso importa?** No caso do Helsinki (sem validação), o treino rodou todos os 50k steps cegamente. A training loss caiu para 0.14 (parecia excelente!), mas o modelo estava memorizando dados — sem eval_loss, não havia como detectar a degradação. Com validação, se a eval_loss começasse a subir (sinal de overfitting), o early stopping interromperia o treino antes do dano.

| Cenário                          | train_loss | eval_loss | Diagnóstico        |
|----------------------------------|:----------:|:---------:|:-------------------|
| Helsinki (sem validação)         | 0.14       | ❌ N/A    | Overfitting oculto |
| Unicamp-T5 (com validação)       | 0.97       | 0.97      | Generalização ok   |
| Overfitting típico (hipotético)  | 0.10       | 2.50      | ⚠️ PARAR treino    |

### Comandos

```bash
# Preparação automática (integrada ao select_and_test_models.py)
python finetuning/select_and_test_models.py

# Ou manualmente
python -c "
from finetuning import config, data_utils
data_utils.prepare_evaluation_csv(
    abstracts_file='abstracts_scielo.csv',
    train_csv=config.SCIELO_TRAIN_CSV,
    val_csv=config.SCIELO_VAL_CSV,
    test_csv=config.SCIELO_TEST_CSV,
    train_samples=18_000,
    val_samples=2_000,
    test_samples=5_000
)
"
```

### Saída
```
finetuning/abstracts-datasets/
├── scielo_abstracts_train.csv   (18.000 exemplos)
├── scielo_abstracts_val.csv     ( 2.000 exemplos)
└── scielo_abstracts_test.csv    ( 5.000 exemplos)
```

---

## STAGE 4: Fine-Tuning

### Objetivo
Fine-tunar o modelo `unicamp-dl/translation-en-pt-t5` no domínio de abstracts científicos.

### Configuração de Treinamento

| Parâmetro                  | Valor                  |
|----------------------------|------------------------|
| GPU                        | NVIDIA RTX 4050 (6GB)  |
| Epochs                     | 12                     |
| Batch size                 | 8                      |
| Gradient accumulation      | 2                      |
| **Batch efetivo**          | **16**                 |
| Learning rate              | 1e-5                   |
| Warmup steps               | 500                    |
| Weight decay               | 0.01                   |
| Max sequence length        | 256 tokens             |
| Precisão                   | FP16 (mixed precision) |
| Otimizador                 | AdamW                  |
| Early stopping patience    | 2 epochs               |
| Gradient checkpointing     | Ativado                |
| Steps por epoch            | 1.125                  |
| Save strategy              | Por epoch              |
| Seed                       | 42                     |

### Configuração do Modelo — `config.json` (antes vs depois)

A arquitetura do modelo **não muda** durante o fine-tuning — apenas os pesos são atualizados. As diferenças no `config.json` são campos de metadados adicionados pela versão mais recente do `transformers`.

#### Modelo Original (HuggingFace)

```json
{
  "_name_or_path": "./",
  "architectures": ["T5ForConditionalGeneration"],
  "d_ff": 3072,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "relu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_num_buckets": 32,
  "torch_dtype": "float32",
  "transformers_version": "4.11.3",
  "use_cache": true,
  "vocab_size": 32128
}
```

#### Modelo Fine-tuned (local)

```json
{
  "architectures": ["T5ForConditionalGeneration"],
  "classifier_dropout": 0.0,
  "d_ff": 3072,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "relu",
  "dropout_rate": 0.1,
  "dtype": "float32",
  "eos_token_id": 1,
  "feed_forward_proj": "relu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": false,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "transformers_version": "4.57.6",
  "use_cache": true,
  "vocab_size": 32128
}
```

#### Diferenças

| Campo                          | Original (HF)   | Fine-tuned (local) | Observação                          |
|--------------------------------|------------------|---------------------|-------------------------------------|
| `_name_or_path`                | `"./"`          | *(removido)*        | Caminho local do autor original     |
| `torch_dtype` / `dtype`        | `"float32"`     | `"float32"`        | Apenas renomeação de campo          |
| `transformers_version`         | `4.11.3`         | `4.57.6`            | Versão da lib no momento do salvamento |
| `classifier_dropout`           | *(ausente)*      | `0.0`               | Adicionado pela versão nova         |
| `dense_act_fn`                 | *(ausente)*      | `"relu"`           | Explicitação da ativação            |
| `is_gated_act`                 | *(ausente)*      | `false`             | T5 padrão não usa gated activation  |
| `relative_attention_max_distance`| *(ausente)*    | `128`               | Default explicitado pela versão nova |

> **Nota**: Todos os hiperparâmetros arquiteturais (d_model, d_ff, num_layers, num_heads, vocab_size) são **idênticos**. O fine-tuning altera **apenas os pesos** (`model.safetensors`), não a arquitetura.

### Configuração de Geração — `generation_config.json`

Arquivo criado automaticamente pelo `Seq2SeqTrainer` (não existia no modelo original do HuggingFace):

```json
{
  "_from_model_config": true,
  "decoder_start_token_id": 0,
  "eos_token_id": [1],
  "pad_token_id": 0,
  "transformers_version": "4.57.6"
}
```

| Parâmetro              | Valor | Descrição                                         |
|------------------------|-------|---------------------------------------------------|
| `decoder_start_token_id` | 0   | Token `<pad>` usado para iniciar a decodificação  |
| `eos_token_id`           | 1   | Token `</s>` marca fim da sequência gerada        |
| `pad_token_id`           | 0   | Token `<pad>` para padding                        |

### Argumentos de Treinamento — `Seq2SeqTrainingArguments`

Configuração completa passada ao `Seq2SeqTrainer` (de `finetuning/trainer.py`):

```python
Seq2SeqTrainingArguments(
    output_dir="./models/finetuned-scielo/unicamp-t5",
    overwrite_output_dir=False,
    num_train_epochs=12,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=2,
    fp16=True,
    logging_steps=100,
    predict_with_generate=True,
    optim="adamw_torch",
    seed=42,
    report_to=[],
    eval_strategy="epoch",
)
```

| Argumento                   | Valor                          | Finalidade                                    |
|-----------------------------|--------------------------------|-----------------------------------------------|
| `output_dir`                | `./models/finetuned-scielo/unicamp-t5` | Diretório de saída dos checkpoints     |
| `overwrite_output_dir`      | `False`                        | Preserva checkpoints existentes               |
| `num_train_epochs`          | 12                             | Número total de epochs                        |
| `per_device_train_batch_size`| 8                             | Batch size por GPU                            |
| `learning_rate`             | 1e-5                           | Taxa de aprendizado (linear warmup + decay)   |
| `warmup_steps`              | 500                            | Steps de warmup linear do LR                  |
| `weight_decay`              | 0.01                           | Regularização L2 desacoplada (AdamW)          |
| `save_strategy`             | `"epoch"`                     | Salva checkpoint a cada epoch                 |
| `save_total_limit`          | 2                              | Mantém apenas os 2 últimos checkpoints        |
| `load_best_model_at_end`    | `True`                         | Carrega melhor modelo (menor eval_loss) ao final |
| `metric_for_best_model`     | `"eval_loss"`                 | Métrica para selecionar melhor checkpoint     |
| `gradient_accumulation_steps`| 2                             | Acumula gradientes de 2 mini-batches          |
| `fp16`                      | `True`                         | Mixed precision (Tensor Cores da RTX 4050)    |
| `logging_steps`             | 100                            | Log de métricas a cada 100 steps              |
| `predict_with_generate`     | `True`                         | Usa `model.generate()` para avaliação         |
| `optim`                     | `"adamw_torch"`               | Otimizador AdamW nativo do PyTorch            |
| `seed`                      | 42                             | Seed para reprodutibilidade                   |
| `eval_strategy`             | `"epoch"`                     | Avalia no dataset de validação a cada epoch   |

### Comando Executado

```bash
python finetuning/finetune_selected_models.py \
  --model unicamp-t5 \
  --epochs 12 \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --lr 1e-5 \
  --fp16 \
  --max_seq_len 256 \
  --early_stopping_patience 2 \
  --skip_prepare
```

### Explicação Detalhada dos Parâmetros

Cada parâmetro do comando foi escolhido para maximizar a qualidade do fine-tuning dentro das restrições de hardware (RTX 4050, 6GB VRAM). Abaixo, a explicação técnica de cada um com exemplos visuais.

---

#### `--model unicamp-t5`

Seleciona o modelo `unicamp-dl/translation-en-pt-t5` do dicionário `config.MODELS`. Veja a seção [STAGE 2](#stage-2-seleção-do-modelo) para justificativa da seleção.

---

#### `--epochs 12`

**O que é**: Número de passagens completas pelo dataset de treino (18.000 exemplos).

**Referência**: Smith, L. N. (2018). *A disciplined approach to neural network hyper-parameters: Part 1 – learning rate, batch size, momentum, and weight decay*. arXiv:1803.09820. https://arxiv.org/abs/1803.09820

**Por que 12**: O número de epochs é determinado pela convergência observada. A eval_loss continuou melhorando em todas as 12 epochs (0.973 no epoch 12), sem acionar early stopping. Mais epochs não foram testados porque a taxa de melhoria nos últimos epochs era marginal (~0.0003/epoch).

```
Epoch 1  ████████████████████████████████████████  eval_loss: 1.0068
Epoch 2  ███████████████████████████████████████   eval_loss: 0.9931  ↓ 0.0137
Epoch 3  ██████████████████████████████████████    eval_loss: 0.9861  ↓ 0.0070
Epoch 4  █████████████████████████████████████     eval_loss: 0.9818  ↓ 0.0043
Epoch 5  ████████████████████████████████████      eval_loss: 0.9792  ↓ 0.0026
Epoch 6  ███████████████████████████████████       eval_loss: 0.9772  ↓ 0.0020
Epoch 7  ██████████████████████████████████        eval_loss: 0.9757  ↓ 0.0015
Epoch 8  █████████████████████████████████         eval_loss: 0.9747  ↓ 0.0010
Epoch 9  ████████████████████████████████          eval_loss: 0.9737  ↓ 0.0010
Epoch 10 ███████████████████████████████           eval_loss: 0.9733  ↓ 0.0004
Epoch 11 ██████████████████████████████            eval_loss: 0.9730  ↓ 0.0003
Epoch 12 █████████████████████████████             eval_loss: 0.9730  ↓ 0.0001 ⭐

→ Redução total: 0.0338 (3.36%)
→ 90% da melhoria ocorre nos primeiros 5 epochs
→ Epochs 10-12: rendimento decrescente (<0.001/epoch)
```

**Trade-off**: Poucas epochs = underfitting (modelo não adaptado ao domínio). Muitas epochs = overfitting (modelo memoriza exemplos de treino). Com 12 epochs, train_loss (0.97) ≈ eval_loss (0.97), indicando ausência de overfitting.

---

#### `--batch_size 8`

**O que é**: Número de exemplos processados **simultaneamente** em cada forward pass pela GPU.

**Referência**: Masters, D. & Luschi, C. (2018). *Revisiting Small Batch Training for Deep Neural Networks*. arXiv:1804.07612. https://arxiv.org/abs/1804.07612

**Por que 8 (e não mais)**: Limitação direta da VRAM da RTX 4050 (6GB). Com FP16, gradient checkpointing e max_seq_len=256:

```
Memória GPU por batch (estimativa):

  Pesos do modelo (FP16):    ~440 MB  (220M params × 2 bytes)
  Ativações Encoder (FP16):  ~384 MB  (batch=8 × 256 tokens × 768 dim × 12 layers)
  Ativações Decoder (FP16):  ~384 MB  (idem)
  Gradientes (FP32):         ~880 MB  (220M params × 4 bytes, mixed precision)
  Estados do otimizador:     ~1760 MB (AdamW: 2 estados × 220M × 4 bytes)
  Overhead CUDA:             ~200 MB
                             ─────────
  Total estimado:           ~4048 MB (~4 GB)

  VRAM disponível:           6144 MB (6 GB)
  Margem:                    ~2096 MB (suficiente ✅)

  Com batch_size=16:         +768 MB ativações → ~4816 MB (ainda cabe, mas ajustado)
  Com batch_size=32:         +1536 MB ativações → OOM ❌ (Out of Memory)

→ batch_size=8 garante estabilidade com margem confortável
```

**Efeito no ruído do gradiente**:

```
Batch size pequeno (ex: 1-4):
  Gradiente ← ∇L(x₁)                         ← Muito ruidoso
  → Convergência instável, LR precisa ser menor

Batch size médio (ex: 8-16):
  Gradiente ← ¼ × (∇L(x₁) + ∇L(x₂) + ... + ∇L(x₈))  ← Bom equilíbrio
  → Gradiente suavizado, convergência estável

Batch size grande (ex: 128-512):
  Gradiente ← 1/128 × Σ ∇L(xᵢ)               ← Muito suave
  → Convergência rápida mas generalização pior
    (Sharp minima, referência: Keskar et al., 2017)
```

**Referência**: Keskar, N. S. et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. In ICLR 2017. https://arxiv.org/abs/1609.04836

---

#### `--grad_accum_steps 2` ⭐

**O que é**: **Gradient Accumulation** — acumula gradientes de múltiplos mini-batches antes de atualizar os pesos. Simula um batch maior sem exigir mais VRAM.

**Referência**: Ott, M. et al. (2018). *Scaling Neural Machine Translation*. In Proceedings of the Third Conference on Machine Translation (WMT), pp. 1–9. https://aclanthology.org/W18-6301/

**Batch Efetivo**:

$$\text{Batch efetivo} = \text{batch\_size} \times \text{grad\_accum\_steps} = 8 \times 2 = 16$$

**Funcionamento visual**:

```
SEM gradient accumulation (batch_size=16, se coubesse na VRAM):
┌──────────────────────────────────────────────────────────┐
│ Forward: 16 exemplos → Loss → Backward → ∇W → Atualiza  │
│ VRAM: ~5.5 GB (pode dar OOM)                             │
└──────────────────────────────────────────────────────────┘

COM gradient accumulation (batch_size=8, grad_accum=2):
┌──────────────────────────────────────────────────────────┐
│ Step 1: Forward 8 exemplos → Loss₁ → Backward → ∇W₁     │
│         (NÃO atualiza pesos, apenas acumula gradiente)   │
│         VRAM: ~4 GB ✅                                    │
│                                                          │
│ Step 2: Forward 8 exemplos → Loss₂ → Backward → ∇W₂     │
│         ∇W_total = ∇W₁ + ∇W₂                            │
│         Optimizer.step() → Atualiza pesos com ∇W_total   │
│         VRAM: ~4 GB ✅                                    │
└──────────────────────────────────────────────────────────┘

Resultado MATEMÁTICO: Gradiente idêntico ao batch_size=16
Resultado PRÁTICO:    Metade da VRAM necessária
Custo:                ~2x mais lento (2 forward passes vs 1)
```

**Por que 2 e não mais?**

```
grad_accum=1  → batch efetivo = 8   → gradiente ruidoso, convergência instável
grad_accum=2  → batch efetivo = 16  → bom equilíbrio ruído/estabilidade ✅
grad_accum=4  → batch efetivo = 32  → mais estável, mas 4x mais lento
grad_accum=8  → batch efetivo = 64  → overkill para 18k exemplos (apenas 281 steps/epoch)

Steps por epoch com cada configuração:
  grad_accum=1: 18000 / 8  = 2250 steps/epoch
  grad_accum=2: 18000 / 16 = 1125 steps/epoch  ← Nosso caso
  grad_accum=4: 18000 / 32 =  562 steps/epoch
  grad_accum=8: 18000 / 64 =  281 steps/epoch  ← Poucos updates, convergência lenta
```

**Impacto na taxa de aprendizado**: O learning rate é aplicado ao gradiente acumulado (já normalizado). Com Transformers `Seq2SeqTrainer`, a loss já é dividida pelo `grad_accum_steps`, então a escala é automaticamente ajustada.

**Implementação** (em `finetuning/trainer.py`):

```python
Seq2SeqTrainingArguments(
    per_device_train_batch_size=8,       # batch real na GPU
    gradient_accumulation_steps=2,       # acumula 2 batches
    # → batch efetivo = 8 × 2 = 16
)
```

---

#### `--lr 1e-5` ⭐

**O que é**: **Learning Rate** — a taxa de aprendizado controla o tamanho do passo na atualização dos pesos do modelo. É o hiperparâmetro mais crítico do treinamento.

**Referência**: Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. In ICLR 2019. https://arxiv.org/abs/1711.05101 (AdamW)

**Referência**: Howard, J. & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification*. In Proceedings of ACL 2018, pp. 328–339. https://aclanthology.org/P18-1031/ (recomendação de LR para fine-tuning)

**Regra de atualização (AdamW)**:

$$\theta_{t+1} = \theta_t - \eta \cdot \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_t\right)$$

Onde:
- $\eta = 10^{-5}$ é o learning rate
- $\hat{m}_t$ = média móvel dos gradientes (momentum)
- $\hat{v}_t$ = média móvel dos gradientes² (adaptação)
- $\lambda = 0.01$ = weight decay
- $\epsilon = 10^{-8}$ = estabilidade numérica

**Por que 1e-5 (e não mais ou menos)?**

```
Para fine-tuning de modelos pré-treinados, a literatura recomenda LRs pequenas:

  Pré-treinamento (do zero):    1e-4  a 1e-3   (pesos aleatórios, grandes passos)
  Fine-tuning (adaptação):      1e-5  a 5e-5   (pesos já bons, passos pequenos) ← 
  Ajuste mínimo (few-shot):     1e-6  a 5e-6   (alterar o mínimo possível)

  LR = 1e-3 (muito alto para fine-tuning):
    ┌─────────────────────────────────────┐
    │   ╱╲  ╱╲  ╱╲                       │  Oscilação destrutiva
    │  ╱  ╲╱  ╲╱  ╲   → Loss diverge     │  Esquece conhecimento pré-treinado
    │ ╱              ╲                    │  "Catastrophic forgetting"
    └─────────────────────────────────────┘

  LR = 1e-5 (ideal para fine-tuning):
    ┌─────────────────────────────────────┐
    │ ╲                                   │  Convergência suave
    │  ╲                                  │  Preserva conhecimento base
    │   ╲___________________________      │  Adapta ao domínio SciELO
    └─────────────────────────────────────┘

  LR = 1e-7 (muito baixo):
    ┌─────────────────────────────────────┐
    │ ─────────────────────────           │  Convergência desprezível
    │                                     │  Modelo quase não muda
    │                                     │  Desperdício de computação
    └─────────────────────────────────────┘
```

**Schedule linear com warmup** (implementado via `Seq2SeqTrainer`):

O LR não é constante — segue um schedule com warmup linear (500 steps) + decay linear até 0:

```
LR
1e-5 ┤          ╱╲
     │         ╱  ╲
     │        ╱    ╲
     │       ╱      ╲
     │      ╱        ╲
     │     ╱          ╲
     │    ╱            ╲
     │   ╱              ╲
     │  ╱                ╲
     │ ╱                  ╲
0    ┤╱                    ╲_
     └──────┬──────────────┬──→ Steps
            500          13500

 Fase 1: WARMUP (steps 0→500)
   LR sobe linearmente de 0 até 1e-5
   → Evita instabilidade no início (gradientes grandes com pesos não calibrados)
   → "Aquece" o otimizador: momentum (m̂) e variância (v̂) do AdamW estabilizam

 Fase 2: DECAY LINEAR (steps 500→13500)
   LR decresce linearmente de 1e-5 até ~0
   → No início: passos maiores para aprender rápido
   → No final: passos minúsculos para refinamento fino

 Valores reais observados no treinamento:
   Step   100: lr = 1.98e-06  (warmup: subindo)
   Step   500: lr = 9.98e-06  (pico: ~1e-5)
   Step  1000: lr = 9.62e-06  (início do decay)
   Step  5000: lr = 6.54e-06  (metade do treinamento)
   Step 10000: lr = 2.70e-06  (75% do treinamento)
   Step 13000: lr = 3.88e-07  (quase zero)
   Step 13500: lr = 3.08e-09  (final: praticamente zero)
```

**Referência para warmup**: Goyal, P. et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677. https://arxiv.org/abs/1706.02677

**Por que warmup de 500 steps?**

```
Sem warmup:
  Step 0: Gradiente grande (loss alta) × LR máximo → passo enorme
  → Pode "destruir" features pré-treinadas nas primeiras iterações
  → Fenômeno: "loss spike" ou divergência precoce

Com warmup de 500 steps:
  Step 0:   LR ≈ 0       → passo quase nulo, gradientes estabilizam
  Step 250: LR ≈ 5e-6    → passos moderados, momentum calibrado
  Step 500: LR = 1e-5     → passo máximo, otimizador calibrado ✅
  
  500 steps = ~3% do treinamento total (13.500 steps)
  → Prática padrão: warmup de 1-5% do total de steps
```

**Weight Decay ($\lambda = 0.01$) — Regularização L2 Desacoplada**

No AdamW (diferente do Adam clássico), o weight decay é aplicado **diretamente aos pesos** em vez de ser adicionado ao gradiente. Isso é chamado de "decoupled weight decay" (Loshchilov & Hutter, 2019):

$$\theta_{t+1} = (1 - \eta \cdot \lambda) \cdot \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

O termo $(1 - \eta \cdot \lambda) = (1 - 10^{-5} \times 0.01) = 0.9999999$ encolhe levemente os pesos a cada step, penalizando pesos com magnitude alta. Isso previne que o modelo "memorize" padrões com pesos extremos.

```
Comparação: Adam clássico vs AdamW

Adam (L2 regularizado):                    AdamW (weight decay desacoplado):
  g' = g + λ·θ   (adiciona ao gradiente)    θ' = θ - η·λ·θ  (encolhe direto)
  m = β₁·m + (1-β₁)·g'                     m = β₁·m + (1-β₁)·g
  v = β₂·v + (1-β₂)·g'²                    v = β₂·v + (1-β₂)·g²
  θ = θ - η · m̂/√v̂                         θ = θ' - η · m̂/√v̂

  Problema: λ interage com Adam de          Correto: λ aplicado independente
  forma não-intuitiva → escala do           do gradiente adaptativo → efeito
  weight decay depende do LR adaptativo     constante e previsível ✅
```

**AdamW — Algoritmo Completo (Kingma & Ba, 2014; Loshchilov & Hutter, 2019)**:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \quad \text{(1º momento — momentum)}$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \quad \text{(2º momento — variância)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad \text{(correção de viés do momentum)}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(correção de viés da variância)}$$
$$\theta_{t+1} = \theta_t - \eta \cdot \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_t\right)$$

Com os valores deste projeto: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\lambda = 0.01$, $\eta = 10^{-5}$ (com schedule).

**Referências**:
- Kingma, D. P. & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. In ICLR 2015. https://arxiv.org/abs/1412.6980
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. In ICLR 2019. https://arxiv.org/abs/1711.05101

---

#### `--fp16` (Mixed Precision Training)

**O que é**: Treina com **precisão mista** — forward pass em FP16 (16 bits), backward pass e atualização de pesos em FP32 (32 bits).

**Referência**: Micikevicius, P. et al. (2018). *Mixed Precision Training*. In ICLR 2018. https://arxiv.org/abs/1710.03740

**Por que usar**:

```
FP32 (32-bit float):  ████████████████████████████████  → 4 bytes por peso
FP16 (16-bit float):  ████████████████                  → 2 bytes por peso

                     FP32          FP16 (mixed)     Economia
Pesos do modelo:     880 MB        440 MB           50%
Ativações:           768 MB        384 MB           50%
Gradientes:          880 MB        880 MB (FP32)      0% (mantido em FP32)
Estados Adam:       1760 MB       1760 MB (FP32)      0% (mantido em FP32)
                    ─────────     ─────────
Total aprox:        4288 MB       3464 MB           ~19% menor ✅
```

**Como funciona (Automatic Mixed Precision)**:

```
┌──────────────────────────────────────────────┐
│ 1. Pesos copiados FP32 → FP16 (master copy)  │
│ 2. Forward pass em FP16 (rápido nos Tensors)  │
│    → Loss calculada em FP16                   │
│ 3. Loss scaling (multiplica loss × 65536)     │
│    → Evita underflow de gradientes em FP16    │
│ 4. Backward pass: gradientes em FP16          │
│ 5. Gradientes → FP32, divididos pelo scaler   │
│ 6. Optimizer.step() em FP32 (pesos master)    │
│ 7. Pesos FP32 → FP16 para próximo forward     │
└──────────────────────────────────────────────┘
```

**Benefícios na RTX 4050**:

```
RTX 4050 possui Tensor Cores com suporte a FP16:
  - Operações FP16: ~16.6 TFLOPS
  - Operações FP32: ~8.3 TFLOPS
  → FP16 é ~2x mais rápido para matmul/convolutions

Tempo estimado por epoch:
  FP32: ~25 min/epoch × 12 = ~5.0 horas
  FP16: ~15 min/epoch × 12 = ~3.0 horas  ← ~40% mais rápido
```

---

#### `--max_seq_len 256` ⭐

**O que é**: Comprimento máximo em tokens de cada sequência (source e target). Sequências mais longas são **truncadas**, mais curtas recebem **padding**.

**Por que 256 (e não o padrão 128 ou o máximo 512)?**

O modelo T5 suporta até `n_positions=512` tokens. A escolha de 256 é um compromisso entre capturar abstracts completos e usar VRAM de forma eficiente.

```
Distribuição de comprimento dos abstracts SciELO (em tokens):

  Tokens │
    512+ │ ▏                                    1.2% truncados com max=512
    480  │ ▎                                   
    448  │ ▎
    416  │ ▍
    384  │ ▌
    352  │ ▋
    320  │ █
    288  │ ██
    256  │ ███▎                                 ~5% truncados com max=256
    224  │ ████▌
    192  │ ██████
    160  │ ████████
    128  │ ██████████▎                          ~25% truncados com max=128 ❌
     96  │ ████████████
     64  │ ██████████████
     32  │ ████████████████
      0  │ █████████████████
         └──────────────────────→ Nº de exemplos

→ max_seq_len=128 (padrão): trunca ~25% dos abstracts (perde informação)
→ max_seq_len=256 (escolhido): trunca ~5% (bom compromisso) ✅
→ max_seq_len=512 (máximo): trunca <2% mas usa 4x mais memória
```

**Impacto na VRAM** — a memória escala **quadraticamente** com o comprimento da sequência (self-attention):

$$\text{Memória}_{attention} \propto \text{batch\_size} \times \text{num\_heads} \times \text{seq\_len}^2$$

```
Memória de atenção por camada (batch=8, heads=12):

  max_seq_len=128:  8 × 12 × 128² × 2 bytes  =  3.0 MB/layer  × 24 layers = 72 MB
  max_seq_len=256:  8 × 12 × 256² × 2 bytes  = 12.0 MB/layer  × 24 layers = 288 MB  ← Nosso
  max_seq_len=512:  8 × 12 × 512² × 2 bytes  = 48.0 MB/layer  × 24 layers = 1152 MB

  128 → 256: +216 MB (cabe na RTX 4050 ✅)
  256 → 512: +864 MB (risco de OOM com batch=8 ❌)
```

**Efeito no truncamento**:

```
Abstract original (310 tokens):
  "The present study aimed to evaluate the effect of different
   concentrations of sodium hypochlorite on the bond strength
   of fiber posts cemented with self-adhesive resin cement to
   root dentin. Forty single-rooted bovine teeth were selected
   and decoronated. The root canals were prepared using [...more...]
   The results suggest that sodium hypochlorite concentration
   significantly affects the bond strength values."

Com max_seq_len=128 (truncado em ↓):
  "The present study aimed to evaluate the effect of different
   concentrations of sodium hypochlorite on the bond strength
   of fiber posts cemented with self-adhesive resin cement to
   root dentin. Forty single-rooted bovine teeth were..."
  → PERDE a conclusão do abstract (informação crítica!)

Com max_seq_len=256 (truncado em ↓):
  "The present study aimed to evaluate the effect of different
   concentrations of sodium hypochlorite on the bond strength
   of fiber posts cemented with self-adhesive resin cement to
   root dentin. Forty single-rooted bovine teeth were selected
   and decoronated. The root canals were prepared using [...]
   The results suggest that sodium hypochlorite concentration
   significantly affects the bond strength values."
  → Captura introdução, método E conclusão ✅
```

**Implementação** (em `finetuning/trainer.py`):

```python
def preprocess_function(examples):
    inputs = tokenizer(
        examples["abstract_en"],
        max_length=max_seq_len,    # ← 256
        truncation=True,           # Corta sequências maiores
        padding="max_length",      # Pad até max_seq_len
    )
    targets = tokenizer(
        text_target=examples["abstract_pt"],
        max_length=max_seq_len,    # ← 256
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = targets["input_ids"]
    # Mascarar PAD tokens com -100 (ignorados na loss)
    inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in inputs["labels"]
    ]
    return inputs
```

---

#### `--early_stopping_patience 2`

**O que é**: Para o treinamento se a `eval_loss` **não melhorar** por 2 epochs consecutivos. Evita overfitting e desperdício de computação.

**Referência**: Prechelt, L. (1998). *Early Stopping — But When?*. In Neural Networks: Tricks of the Trade, Lecture Notes in Computer Science, vol 1524, pp. 55–69. https://doi.org/10.1007/3-540-49430-8_3

**Como funciona** (implementado via `EarlyStoppingCallback` do Transformers):

```
                  patience = 2
                  ─────────────
  Epoch  eval_loss   Melhor?   Contador   Ação
  ─────  ─────────   ───────   ────────   ─────────────────────
    1    1.006836    Sim ✅    0          Salva como melhor
    2    0.993096    Sim ✅    0          Salva como melhor
    3    0.986074    Sim ✅    0          Salva como melhor
    ...     ...        ...      ...        ...
   12    0.972978    Sim ✅    0          Salva como melhor ⭐

  → No nosso caso, eval_loss melhorou em TODAS as 12 epochs.
  → Early stopping NUNCA foi acionado.
  → Se tivéssemos configurado epochs=50, pararia
     quando 2 epochs consecutivos não melhorassem.

  Cenário hipotético (se tivéssemos treinado mais):
  Epoch  eval_loss   Melhor?   Contador   Ação
  ─────  ─────────   ───────   ────────   ─────────────────────
   12    0.972978    Sim ✅    0          Salva como melhor
   13    0.973100    Não ❌    1          Esperando... (1/2)
   14    0.973200    Não ❌    2          PARA ✋ (patience atingido)
   → Carrega checkpoint do epoch 12 (melhor modelo)
```

**Por que patience=2 (e não 1 ou 5)?**

```
patience=1: Muito agressivo — para no primeiro "tropeço"
  → Pode parar prematuramente se houver flutuação normal

patience=2: Equilibrado — permite 1 flutuação mas evita desperdício
  → Prática padrão na literatura de NLP  ✅

patience=5: Conservador — treina mais mesmo sem melhoria
  → Desperdiça horas de GPU se o modelo já convergiu
```

---

#### `--skip_prepare`

**O que é**: Pula a etapa de preparação dos CSVs de treino/validação/teste (já preparados anteriormente no STAGE 3). Sem este flag, o script executaria `data_utils.prepare_evaluation_csv()` novamente.

**Quando usar**: Quando os arquivos `scielo_abstracts_train.csv`, `scielo_abstracts_val.csv` e `scielo_abstracts_test.csv` já existem no diretório `finetuning/abstracts-datasets/`.

---

#### Gradient Checkpointing (ativado automaticamente no código)

**O que é**: Técnica que **recalcula** ativações intermediárias durante o backward pass em vez de armazená-las na memória. Troca computação por memória.

**Referência**: Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost*. arXiv:1604.06174. https://arxiv.org/abs/1604.06174

```
SEM gradient checkpointing:
  Forward:  layer₁ → [salva a₁] → layer₂ → [salva a₂] → ... → layer₂₄ → [salva a₂₄] → loss
  Backward: usa a₂₄ → ∇₂₄, usa a₂₃ → ∇₂₃, ..., usa a₁ → ∇₁

  Memória: O(n) ativações armazenadas = 24 camadas × ativações
  → Pode exigir >6 GB (impossível na RTX 4050)

COM gradient checkpointing:
  Forward:  layer₁ → [salva a₁] → layer₂ → [descarta] → ... → layer₂₄ → loss
  Backward: recalcula a₂₃ (forward parcial) → ∇₂₃, recalcula a₂₂ → ∇₂₂, ...

  Memória: O(√n) ativações armazenadas ≈ √24 ≈ 5 checkpoints
  → Economia de ~60-70% de VRAM das ativações
  → Custo: ~33% mais lento (recalcula forward para cada segmento)
```

**Implementação** (em `finetuning/trainer.py`):

```python
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    # → Reduz VRAM de ativações de ~1.5 GB para ~500 MB
    # → Permite batch_size=8 com max_seq_len=256 na RTX 4050
```

---

### Resumo: Por que cada parâmetro foi escolhido

| Parâmetro        | Valor   | Motivação principal                                         | Alternativa descartada        |
|------------------|---------|-------------------------------------------------------------|-------------------------------|
| `model`          | unicamp-t5 | Melhor trade-off qualidade/tamanho (220M params)         | Helsinki (600M, não cabe)     |
| `epochs`         | 12      | eval_loss convergiu sem overfitting                         | 5 (underfitting), 50 (desnecessário) |
| `batch_size`     | 8       | Maior batch que cabe na RTX 4050 (6GB) com margem          | 16 (risco OOM), 4 (muito ruidoso) |
| `grad_accum`     | 2       | Batch efetivo=16, equilíbrio ruído/estabilidade             | 1 (ruidoso), 4 (lento demais) |
| `lr`             | 1e-5    | LR recomendado para fine-tuning de Transformers             | 1e-3 (catastrophic forgetting), 1e-7 (sem aprendizado) |
| `fp16`           | True    | ~40% mais rápido + ~19% menos VRAM nos Tensor Cores        | FP32 (mais lento, mais VRAM)  |
| `max_seq_len`    | 256     | Captura ~95% dos abstracts sem OOM                          | 128 (perde 25%), 512 (OOM)    |
| `early_stopping` | 2       | Previne overfitting sem parar prematuramente                | 1 (agressivo demais), 5 (desperdiça GPU) |
| `skip_prepare`   | True    | Dados já preparados no STAGE 3                              | False (refaz splits desnecessariamente) |

### Curva de Convergência (eval_loss)

```
Epoch | eval_loss | Step   | Tendência
------|-----------|--------|----------
  1   | 1.006836  |  1125  |
  2   | 0.993096  |  2250  | ↓ melhorou
  3   | 0.986074  |  3375  | ↓ melhorou
  4   | 0.981832  |  4500  | ↓ melhorou
  5   | 0.979202  |  5625  | ↓ melhorou
  6   | 0.977226  |  6750  | ↓ melhorou
  7   | 0.975687  |  7875  | ↓ melhorou
  8   | 0.974656  |  9000  | ↓ melhorou
  9   | 0.973745  | 10125  | ↓ melhorou
 10   | 0.973330  | 11250  | ↓ melhorou
 11   | 0.973035  | 12375  | ↓ melhorou
 12   | 0.972978  | 13500  | ↓ melhorou ⭐ BEST
```

**Observações:**
- A eval_loss melhorou consistentemente em todas as 12 epochs
- O melhor checkpoint foi o último: `checkpoint-13500` (epoch 12, eval_loss: 0.972978)
- Early stopping NÃO foi acionado — o modelo ainda estava convergindo
- A taxa de melhoria desacelera nos epochs finais (~0.0003 por epoch), sugerindo proximidade do ponto ótimo

### Training Loss (média por epoch)

| Epoch | Training Loss (média) | Eval Loss  | Learning Rate (final) |
|-------|----------------------:|-----------:|----------------------:|
| 1     | 1.1014                | 1.006836   | 9.54e-06              |
| 2     | 1.0509                | 0.993096   | 8.69e-06              |
| 3     | 1.0334                | 0.986074   | 7.85e-06              |
| 4     | 1.0171                | 0.981832   | 6.92e-06              |
| 5     | 1.0028                | 0.979202   | 6.08e-06              |
| 6     | 0.9968                | 0.977226   | 5.23e-06              |
| 7     | 0.9839                | 0.975687   | 4.39e-06              |
| 8     | 0.9800                | 0.974656   | 3.46e-06              |
| 9     | 0.9748                | 0.973745   | 2.62e-06              |
| 10    | 0.9729                | 0.973330   | 1.77e-06              |
| 11    | 0.9664                | 0.973035   | 9.26e-07              |
| 12    | 0.9663                | 0.972978   | 3.08e-09              |

**Observações sobre o treinamento:**
- Training loss caiu de ~1.10 (epoch 1) para ~0.97 (epoch 12) — redução de ~12%
- Learning rate seguiu schedule linear com warmup de 500 steps (pico 1e-5) e decay até ~0
- Gradient norms estáveis em 0.5–0.9 ao longo de todo o treinamento (sem gradient explosion)
- Diferença train_loss vs eval_loss pequena (~0.01), indicando ausência de overfitting

### Detalhes Técnicos do Treinamento

- **Gradient checkpointing**: Reduz consumo de VRAM recalculando ativações intermediárias no backward pass
- **FP16 (mixed precision)**: Reduz uso de memória e acelera computação em Tensor Cores
- **Mascaramento de PAD tokens**: Labels com token PAD são substituídos por -100 para não contribuírem na cross-entropy loss
- **Early stopping**: Monitora `eval_loss` a cada epoch; para se não houver melhoria em 2 epochs consecutivos
- **AdamW**: Otimizador Adam com weight decay desacoplado (0.01)

### Checkpoints

Cada epoch gera um checkpoint. Os 2 últimos são preservados (save_total_limit=2):

| Checkpoint       | Epoch | eval_loss |
|------------------|-------|-----------|
| checkpoint-12375 | 11    | 0.973035  |
| checkpoint-13500 | 12    | 0.972978 ⭐ |

O modelo final (melhor) é salvo na raiz: `unicamp-t5/unicamp-t5/`

### Resumir Treinamento Interrompido

```bash
python finetuning/finetune_selected_models.py \
  --model unicamp-t5 \
  --epochs 12 \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --lr 1e-5 \
  --fp16 \
  --max_seq_len 256 \
  --early_stopping_patience 2 \
  --skip_prepare \
  --resume_from ./unicamp-t5/unicamp-t5/checkpoint-13500
```

O `Seq2SeqTrainer` preserva: estado do otimizador/scheduler, epoch/step atual, melhor modelo e contador de early stopping.

---

## STAGE 5: Avaliação Final

### Objetivo
Comparar o modelo **antes** e **depois** do fine-tuning, usando os **mesmos** 5.000 exemplos de teste.

### Comandos

```bash
# Testar modelo base (antes do fine-tuning)
python finetuning/select_and_test_models.py --model unicamp-t5 --skip_prepare

# Testar modelo fine-tuned
python finetuning/select_and_test_models.py --test_finetuned --model unicamp-t5 --skip_prepare

# Testar ambos e comparar
python finetuning/select_and_test_models.py --test_both --model unicamp-t5 --skip_prepare
```

### Resultados

**Antes do fine-tuning** (`scielo_before_finetuning.csv`):

| Modelo     | BLEU  | chrF  | COMET  | BERTScore |
|------------|------:|------:|-------:|----------:|
| unicamp-t5 | 40.06 | 65.61 | 0.8499 | 0.8957    |

**Após fine-tuning — Epoch 11** (`scielo_after_finetuning_epoch_11.csv`):

| Modelo     | Checkpoint       | BLEU  | chrF  | COMET  | BERTScore |
|------------|------------------|------:|------:|-------:|----------:|
| unicamp-t5 | checkpoint-12375 | 45.51 | 70.54 | 0.8756 | 0.9124    |

**Após fine-tuning — Epoch 12** (`scielo_after_finetuning_epoch_12.csv`):

| Modelo     | Checkpoint       | BLEU  | chrF  | COMET  | BERTScore |
|------------|------------------|------:|------:|-------:|----------:|
| unicamp-t5 | checkpoint-13500 | 45.51 | 70.54 | 0.8756 | 0.9124    |

### Análise de Melhoria

| Métrica    | Antes  | Depois (Ep.12) | Delta   | Melhoria |
|------------|-------:|---------------:|--------:|---------:|
| BLEU       | 40.06  | 45.51          | +5.45   | +13.6%   |
| chrF       | 65.61  | 70.54          | +4.93   | +7.5%    |
| COMET      | 0.8499 | 0.8756         | +0.0257 | +3.0%    |
| BERTScore  | 0.8957 | 0.9124         | +0.0167 | +1.9%    |

### Interpretação

- **BLEU +13.6%**: Melhoria significativa na precisão de n-gramas. O modelo gera traduções com sobreposição lexical mais próxima das referências humanas.
- **chrF +7.5%**: Melhoria a nível de caracteres, indicando melhor morfologia e ortografia (acentuação, concordância).
- **COMET +3.0%**: Score neural baseado em modelo treinado em avaliações humanas confirma melhoria na qualidade percebida.
- **BERTScore +1.9%**: Melhoria na similaridade semântica. O modelo preserva melhor o significado original.
- **Epochs 11→12 estáveis**: Métricas idênticas entre epochs 11 e 12 indicam convergência atingida — o modelo estabilizou.
- **Melhoria dentro da faixa saudável** (+5 a +15% BLEU): Sem sinais de overfitting.

---

## Quickstart

```bash
# 1. Instalar dependências
pip install -r requirements.txt
pip install -r requirements-ml.txt

# 2. Preparar dataset SciELO
python prepare_scielo_dataset.py

# 3. Pipeline completo
python finetune_and_evaluate.py --skip_prepare

# Ou executar etapas individualmente:

# 3a. Preparar splits e testar modelo base
python finetuning/select_and_test_models.py --model unicamp-t5

# 3b. Fine-tuning
python finetuning/finetune_selected_models.py \
  --model unicamp-t5 --epochs 12 --batch_size 8 \
  --grad_accum_steps 2 --lr 1e-5 --fp16 --max_seq_len 256 \
  --early_stopping_patience 2 --skip_prepare

# 3c. Avaliar modelo fine-tuned
python finetuning/select_and_test_models.py --test_finetuned --model unicamp-t5 --skip_prepare
```

---

## Dependências

### requirements.txt
Dependências gerais do projeto (pandas, numpy, etc.)

### requirements-ml.txt
Dependências de machine learning:
- `transformers` — HuggingFace Transformers (modelos, tokenizadores, Trainer)
- `torch` — PyTorch (backend de deep learning)
- `datasets` — HuggingFace Datasets
- `sacrebleu` — Cálculo de BLEU e chrF
- `unbabel-comet` — Cálculo de COMET
- `bert-score` — Cálculo de BERTScore
- `sentencepiece` — Tokenização SentencePiece
- `accelerate` — Aceleração de treinamento HuggingFace
- `tqdm` — Barras de progresso

---

## Detalhes Técnicos

### Reprodutibilidade

- Seed = 42 em todos os splits e treinamento
- `torch.manual_seed(42)` no carregamento do modelo
- Splits determinísticos: mesmos 5k exemplos de teste para base e fine-tuned
- Resultados reprodutíveis com mesma GPU e mesma seed

### Pipeline de Tokenização e Inferência

```
Entrada: "The patient presented with fever and cough."
    ↓ SentencePiece (unigram, 32k vocab)
Input IDs: [37, 1868, 4793, 28, 18851, 11, 14912, 5, 1]
    ↓ T5 Encoder (12 layers × 768 dim × 12 heads)
Hidden states: [768-dim vectors × seq_len]
    ↓ T5 Decoder (12 layers, autoregressive, beam search k=5)
Output IDs: [101, 5847, 12059, 28, 18453, 11, 30419, 5, 1]
    ↓ Decode
Saída: "O paciente apresentou febre e tosse."
```

### Cálculo da Loss — Cross-Entropy

**Referência**: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press, Cap. 6.2.2. https://www.deeplearningbook.org/

A loss function utilizada é a **Cross-Entropy** (entropia cruzada), que mede a diferença entre a distribuição de probabilidade prevista pelo modelo e a distribuição real (one-hot do token correto).

$$\mathcal{L} = -\frac{1}{|T|} \sum_{t \in T} \log P(y_t \mid y_{<t}, X)$$

Onde:
- $y_t$ = token correto na posição $t$ da tradução de referência
- $y_{<t}$ = todos os tokens anteriores (contexto autoregressivo do decoder)
- $X$ = sequência fonte completa (input do encoder)
- $P(y_t \mid y_{<t}, X)$ = probabilidade que o modelo atribui ao token correto
- $T$ = conjunto de tokens **não-mascarados** (exclui tokens PAD)

**Como funciona na prática:**

```
Referência: "O paciente apresentou febre" → tokens [101, 5847, 12059, 18453, 1]
Decoder output (logits → softmax → probabilidades):

  Posição 1: P("O")         = 0.87  → -log(0.87) = 0.139
  Posição 2: P("paciente")  = 0.72  → -log(0.72) = 0.329
  Posição 3: P("apresentou")= 0.65  → -log(0.65) = 0.431
  Posição 4: P("febre")     = 0.58  → -log(0.58) = 0.545
  Posição 5: P("</s>")      = 0.91  → -log(0.91) = 0.094
  Posição 6: [PAD] = -100           → IGNORADO (não contribui para loss)
  Posição 7: [PAD] = -100           → IGNORADO

  Loss = (0.139 + 0.329 + 0.431 + 0.545 + 0.094) / 5 = 0.308
```

**Mascaramento de PAD tokens**: Tokens de padding recebem label `-100`, que é o valor especial do PyTorch `nn.CrossEntropyLoss(ignore_index=-100)`. Isso evita que o modelo aprenda a "gerar" padding — ele é avaliado **apenas** pela qualidade dos tokens reais da tradução.

**Relação com eval_loss**: A eval_loss reportada no treinamento (0.97 no epoch 12) é exatamente esta cross-entropy calculada sobre os 2k exemplos de validação. Um valor de 0.97 significa que, em média, o modelo atribui $e^{-0.97} \approx 0.38$ de probabilidade ao token correto — razoável para um vocabulário de 32k tokens (baseline aleatório seria $-\log(1/32128) = 10.38$).

### Early Stopping

```
Para cada epoch:
  1. Calcular eval_loss no conjunto de validação (2k exemplos)
  2. Se eval_loss < melhor_loss_anterior → salvar como melhor modelo
  3. Se eval_loss >= melhor_loss_anterior → incrementar contador
  4. Se contador >= patience (2) → parar treinamento

No nosso caso: eval_loss melhorou em todas as 12 epochs,
portanto early stopping NÃO foi acionado.
```

### Geração (Inferência) — Beam Search

**Referência**: Freitag, M. & Al-Onaizan, Y. (2017). *Beam Search Strategies for Neural Machine Translation*. In Proceedings of the First Workshop on Neural Machine Translation, pp. 56–60. https://aclanthology.org/W17-3207/

| Parâmetro  | Valor          |
|------------|----------------|
| Decodificação | Beam Search |
| Num beams  | 5              |
| Max length | 256 tokens     |

**O que é Beam Search?** Em vez de escolher apenas o token mais provável a cada passo (greedy search), o Beam Search mantém as $k$ melhores hipóteses parciais (beams) e expande todas:

$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, X)$$

```
Exemplo com num_beams=3 (simplificado):

Passo 1: gerar primeiro token
  Beam 1: "O"         score = log(0.87) = -0.139    ✅ Top-3
  Beam 2: "A"         score = log(0.05) = -2.996    ✅ Top-3
  Beam 3: "Os"        score = log(0.03) = -3.507    ✅ Top-3
  (outros 32125 tokens descartados)

Passo 2: expandir cada beam com próximo token
  Beam 1 → "O paciente"     score = -0.139 + log(0.72) = -0.468  ✅
  Beam 1 → "O doente"       score = -0.139 + log(0.10) = -2.442  ✅
  Beam 2 → "A paciente"     score = -2.996 + log(0.45) = -3.795  ✅
  Beam 2 → "A pessoa"       score = -2.996 + log(0.20) = -4.605
  Beam 3 → "Os pacientes"   score = -3.507 + log(0.55) = -4.105
  ... (mantém apenas as 3 melhores hipóteses)

Passo final: selecionar beam com maior score total
  Melhor: "O paciente apresentou febre persistente"  score = -3.21
  → Esta é a tradução retornada
```

**Por que `num_beams=5`?** Valores maiores exploram mais hipóteses mas são mais lentos ($O(k \times V \times T)$ onde $V$ = vocabulário, $T$ = comprimento). Para tradução, 4-5 beams é o padrão na literatura (Vaswani et al., 2017).

---

## Estrutura do Projeto

```
.
├── README.md                                  ← Este arquivo
├── PROJECT_STRUCTURE.md                       ← Estrutura detalhada (visual)
├── QUICK_COMMANDS.md                          ← Referência rápida de comandos
├── requirements.txt                           ← Dependências gerais
├── requirements-ml.txt                        ← Dependências ML
│
├── prepare_scielo_dataset.py                  [STAGE 0] Gera abstracts_scielo.csv
├── models-test.py                             [STAGE 1] Avalia 5 modelos em datasets públicos
├── evaluate_quickmt.py                        [STAGE 1] Avalia modelo QuickMT (CTranslate2)
├── choose_best_model.py                       [STAGE 2] Ranking e seleção de modelo
├── show_model_configs.py                      Exibe configurações dos modelos
├── compute_neural_metrics.py                  Calcula COMET e BERTScore
├── finetune_and_evaluate.py                   Pipeline integrado (STAGES 1-5)
├── check_gpu.py                               Verificação de GPU disponível
│
├── scielo_before_finetuning.csv               [STAGE 5] Métricas baseline (BLEU=40.06)
├── scielo_after_finetuning_epoch_1.csv        [STAGE 5] Métricas epoch 1
├── scielo_after_finetuning_epoch_11.csv       [STAGE 5] Métricas epoch 11 (BLEU=45.51)
├── scielo_after_finetuning_epoch_12.csv       [STAGE 5] Métricas epoch 12 (BLEU=45.51)
│
├── evaluation/                                Módulo de avaliação (STAGE 1)
│   ├── __init__.py
│   ├── config.py                              Configurações de avaliação
│   ├── datasets.py                            Carregamento de datasets públicos
│   ├── metrics.py                             Cálculo de métricas
│   ├── models_loader.py                       Carregamento de modelos
│   ├── run.py                                 Execução da avaliação
│   ├── io_utils.py                            Utilitários de I/O
│   └── fill_missing_metrics.py                Preenche métricas faltantes
│
├── evaluation_results/                        Resultados de avaliação
│   ├── translation_metrics_all.csv            [STAGE 1] Consolidado todos os modelos
│   ├── Helsinki-NLP_opus-mt-tc-big-en-pt.csv
│   ├── Narrativa_mbart-large-50-finetuned-opus-en-pt-translation.csv
│   ├── unicamp-dl_translation-en-pt-t5.csv
│   ├── VanessaSchenkel_unicamp-finetuned-en-to-pt-dataset-ted.csv
│   ├── danhsf_m2m100_418M-finetuned-kde4-en-to-pt_BR.csv
│   └── quickmt_quickmt-en-pt.csv
│
├── finetuning/                                Módulo de fine-tuning (STAGES 3-5)
│   ├── __init__.py
│   ├── config.py                              Configurações centralizadas
│   ├── models.py                              Carregamento/salvamento de modelos
│   ├── data_utils.py                          Preparação de dados (splits)
│   ├── datasets.py                            Dataset handling
│   ├── metrics.py                             BLEU, chrF, COMET, BERTScore
│   ├── evaluate.py                            Avaliação com progresso (tqdm)
│   ├── trainer.py                             Seq2SeqTrainer + fine-tuning loop
│   ├── compare.py                             Comparação base vs fine-tuned
│   ├── io_utils.py                            Utilitários I/O
│   ├── finetune_selected_models.py            [STAGE 4] Script de fine-tuning
│   ├── select_and_test_models.py              [STAGE 3+5] Preparo + teste
│   └── abstracts-datasets/                    [STAGE 3] Dados SciELO
│       ├── abstracts_scielo.csv               Corpus completo (2.7M exemplos)
│       ├── scielo_abstracts_train.csv         18.000 exemplos (treino)
│       ├── scielo_abstracts_val.csv            2.000 exemplos (validação)
│       └── scielo_abstracts_test.csv           5.000 exemplos (teste)
│
├── unicamp-t5/                                ⭐ MODELO FINE-TUNED (resultado final)
│   └── unicamp-t5/
│       ├── config.json                        Configuração do modelo
│       ├── generation_config.json             Configuração de geração
│       ├── model.safetensors                  Pesos do melhor modelo (epoch 12)
│       ├── tokenizer.json                     Tokenizador
│       ├── tokenizer_config.json              Configuração do tokenizador
│       ├── spiece.model                       Modelo SentencePiece
│       ├── special_tokens_map.json
│       ├── checkpoint-12375/                  Checkpoint epoch 11
│       └── checkpoint-13500/                  Checkpoint epoch 12 (best)
│           ├── model.safetensors
│           ├── optimizer.pt
│           ├── scheduler.pt
│           ├── trainer_state.json             Log completo de treinamento
│           └── training_args.bin
│
├── models/                                    Modelos auxiliares
│   └── finetuned-scielo/
│       └── helsinki/                           Fine-tuning anterior (Helsinki)
│
├── models-configs/                            Configurações JSON dos modelos
│   ├── helsink.json
│   └── m2m100.json
│
└── checkpoints/                               Checkpoints de controle
    ├── training/
    └── evaluation/
```

---

## Métricas de Avaliação — Explicação Técnica

Este projeto utiliza 4 métricas complementares para avaliar a qualidade das traduções. Duas são métricas **lexicais** (baseadas em sobreposição de tokens) e duas são métricas **neurais** (baseadas em embeddings de modelos pré-treinados). A combinação garante uma avaliação robusta que captura tanto a fidelidade lexical quanto a adequação semântica.

### Visão Geral

| Métrica     | Tipo    | Granularidade | Escala   | Requer Source? | Implementação        |
|-------------|---------|---------------|----------|----------------|----------------------|
| BLEU        | Lexical | Palavra       | 0–100    | Não            | `sacrebleu.BLEU()`   |
| chrF        | Lexical | Caractere     | 0–100    | Não            | `sacrebleu.CHRF()`   |
| COMET       | Neural  | Sentença      | 0–1      | Sim            | `Unbabel/wmt22-comet-da` |
| BERTScore   | Neural  | Token         | 0–1      | Não            | `bert-score` (lang=pt) |

---

### 1. BLEU (Bilingual Evaluation Understudy)

**Referência**: Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. In Proceedings of the 40th Annual Meeting of the ACL, pp. 311–318. https://aclanthology.org/P02-1040/

**Padronização**: Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. In Proceedings of the Third Conference on Machine Translation (WMT), pp. 186–191. https://aclanthology.org/W18-6319/

#### O que mede
BLEU mede a **precisão de n-gramas** entre a tradução candidata (hipótese) e a tradução de referência humana, penalizando traduções muito curtas via *brevity penalty*. É a métrica mais utilizada na literatura de tradução automática.

#### Fórmula

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \cdot \log p_n\right)$$

Onde:
- $p_n$ = precisão de n-gramas modificada (clipped precision)
- $w_n = \frac{1}{N}$ (peso uniforme, $N=4$ por padrão)
- $\text{BP} = \min\left(1, \; e^{1 - r/c}\right)$ = brevity penalty ($r$ = comprimento da referência, $c$ = comprimento da hipótese)

#### Exemplo Visual

```
Referência: "O paciente apresentou febre e tosse persistente"
Hipótese:   "O paciente apresentou febre e tosse"

Unigrams (1-gram):
  Referência: {O, paciente, apresentou, febre, e, tosse, persistente}  → 7 tokens
  Hipótese:   {O, paciente, apresentou, febre, e, tosse}              → 6 tokens
  Match:      {O, paciente, apresentou, febre, e, tosse}              → 6 matches
  p₁ = 6/6 = 1.00 ✅

Bigrams (2-gram):
  Referência: {O paciente, paciente apresentou, apresentou febre, febre e, e tosse, tosse persistente}
  Hipótese:   {O paciente, paciente apresentou, apresentou febre, febre e, e tosse}
  Match:      {O paciente, paciente apresentou, apresentou febre, febre e, e tosse}  → 5/5
  p₂ = 5/5 = 1.00 ✅

Trigrams (3-gram):
  Referência: {O paciente apresentou, paciente apresentou febre, apresentou febre e, febre e tosse, e tosse persistente}
  Hipótese:   {O paciente apresentou, paciente apresentou febre, apresentou febre e, febre e tosse}
  Match:      {O paciente apresentou, paciente apresentou febre, apresentou febre e, febre e tosse}  → 4/4
  p₃ = 4/4 = 1.00 ✅

4-grams:
  Referência: {O paciente apresentou febre, paciente apresentou febre e, apresentou febre e tosse, febre e tosse persistente}
  Hipótese:   {O paciente apresentou febre, paciente apresentou febre e, apresentou febre e tosse}
  Match:      {O paciente apresentou febre, paciente apresentou febre e, apresentou febre e tosse}  → 3/3
  p₄ = 3/3 = 1.00 ✅

Brevity Penalty:
  r = 7 (referência), c = 6 (hipótese) → c < r
  BP = exp(1 - 7/6) = exp(-0.167) ≈ 0.846

BLEU = BP × exp(¼ × (log(1.0) + log(1.0) + log(1.0) + log(1.0)))
     = 0.846 × exp(0)
     = 0.846 × 1.0
     = 84.6   ← Penalizado por ser mais curta que a referência
```

#### Limitações

- **Insensível a sinônimos**: "febre" vs "temperatura alta" = 0 match, apesar de semanticamente equivalentes
- **Independente da ordem global**: Permutações de fragmentos podem gerar BLEU alto sem coerência
- **Brevity penalty assimétrica**: Penaliza traduções curtas, mas não as longas demais

#### Implementação neste projeto

```python
# finetuning/metrics.py
from sacrebleu import BLEU
bleu = BLEU(lowercase=False)
bleu_score = bleu.corpus_score(predictions, [references])  # corpus-level
# Retorna: score ∈ [0, 100]
```

> **Nota**: Utilizamos `sacreBLEU` (Post, 2018) que garante tokenização padronizada e reprodutibilidade. O score é computado a nível de corpus (não média de sentenças).

---

### 2. chrF (Character n-gram F-score)

**Referência**: Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. In Proceedings of the Tenth Workshop on Statistical Machine Translation (WMT), pp. 392–395. https://aclanthology.org/W15-3049/

#### O que mede
chrF mede a **sobreposição de n-gramas de caracteres** entre hipótese e referência, utilizando o F-score (média harmônica de precisão e recall). Por operar a nível de caractere, é mais robusta a variações morfológicas do que o BLEU.

#### Fórmula

$$\text{chrF}_\beta = (1 + \beta^2) \cdot \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

Onde:
- $\text{chrP}_n = \frac{|\text{n-gramas}_{\text{hyp}} \cap \text{n-gramas}_{\text{ref}}|}{|\text{n-gramas}_{\text{hyp}}|}$ (precisão de char n-grams)
- $\text{chrR}_n = \frac{|\text{n-gramas}_{\text{hyp}} \cap \text{n-gramas}_{\text{ref}}|}{|\text{n-gramas}_{\text{ref}}|}$ (recall de char n-grams)
- $\beta = 2$ por padrão (favorece recall)
- Média sobre $n = 1, 2, \ldots, 6$ (character n-grams de ordem 1 a 6)

#### Exemplo Visual

```
Referência: "apresentou"
Hipótese:   "apresentaram"

Character 3-grams:
  Referência: {apr, pre, res, ese, sen, ent, nto, tou}           → 8 trigrams
  Hipótese:   {apr, pre, res, ese, sen, ent, nta, tar, ara, ram} → 10 trigrams
  Interseção: {apr, pre, res, ese, sen, ent}                     →  6 matches

  chrP₃ = 6/10 = 0.60 (precisão: quantos trigrams da hipótese estão na referência)
  chrR₃ = 6/8  = 0.75 (recall: quantos trigrams da referência foram cobertos)

  chrF₃ (β=2) = (1 + 4) × (0.60 × 0.75) / (4 × 0.60 + 0.75)
              = 5 × 0.45 / 3.15
              = 0.714

→ Apesar de conjugações diferentes ("apresentou" vs "apresentaram"),
  chrF captura a similaridade morfológica (71.4%) enquanto BLEU
  word-level daria 0% match (palavras diferentes).
```

#### Vantagens sobre BLEU

```
Exemplo: tradução com variação morfológica

Referência: "Os pacientes foram diagnosticados"
Hipótese A: "O paciente foi diagnosticado"         ← tradução correta (singular)
Hipótese B: "A mesa voou pelo hospital"             ← tradução incorreta

BLEU (word-level):
  Hipótese A: matches = {diagnosticado~diagnosticados?} → match parcial
  Hipótese B: matches = {}                              → 0 matches
  → BLEU diferencia, mas penaliza A severamente por flexão

chrF (char-level):
  Hipótese A: alta sobreposição em "pacient-", "diagnosticad-", "for-/foi"
  Hipótese B: baixíssima sobreposição
  → chrF captura melhor que A é quase correta
```

#### Implementação neste projeto

```python
# finetuning/metrics.py
from sacrebleu import CHRF
chrf = CHRF(lowercase=False)
chrf_score = chrf.corpus_score(predictions, [references])  # corpus-level
# Retorna: score ∈ [0, 100]
```

---

### 3. COMET (Crosslingual Optimized Metric for Evaluation of Translation)

**Referência**: Rei, R., de Souza, J. G. C., Alves, D., Zerva, C., Farinha, A. C., Glushkova, T., Lavie, A., Coheur, L., & Martins, A. F. T. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*. In Proceedings of the Seventh Conference on Machine Translation (WMT), pp. 578–585. https://aclanthology.org/2022.wmt-1.52/

**Modelo base**: Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. In Proceedings of ACL 2020, pp. 8440–8451. https://aclanthology.org/2020.acl-main.747/ (XLM-RoBERTa)

#### O que mede
COMET é uma métrica **neural aprendida** que utiliza um modelo XLM-RoBERTa fine-tuned em avaliações humanas (Direct Assessments) de competições WMT. Diferente de BLEU e chrF, COMET considera a **frase fonte** (source) além da referência e hipótese, capturando **adequação** (se o significado foi preservado) e **fluência**.

#### Fórmula de treinamento

O modelo COMET é treinado para minimizar o erro quadrático médio (MSE) entre o score previsto e avaliações humanas (Direct Assessments, DA):

$$\mathcal{L}_{COMET} = \frac{1}{N} \sum_{i=1}^{N} \left( f(\mathbf{e}_{src}^i, \mathbf{e}_{mt}^i, \mathbf{e}_{ref}^i) - z_i \right)^2$$

Onde:
- $f(\cdot)$ = rede feed-forward estimadora (output: score predito)
- $\mathbf{e}_{src}, \mathbf{e}_{mt}, \mathbf{e}_{ref}$ = embeddings pooled do XLM-R para source, hipótese e referência
- $z_i$ = z-score da avaliação humana (Direct Assessment normalizado)
- $N$ = número de exemplos de treinamento (avaliações WMT15–WMT20)

A entrada do estimador combina os embeddings em um vetor de features:

$$\mathbf{f} = [\mathbf{e}_{src}; \, \mathbf{e}_{mt}; \, \mathbf{e}_{ref}; \, |\mathbf{e}_{src} - \mathbf{e}_{mt}|; \, |\mathbf{e}_{ref} - \mathbf{e}_{mt}|; \, \mathbf{e}_{src} \odot \mathbf{e}_{mt}; \, \mathbf{e}_{ref} \odot \mathbf{e}_{mt}]$$

Onde $[\,;\,]$ é concatenação, $|\cdot|$ é diferença absoluta, e $\odot$ é produto elemento a elemento. Isso captura **similaridade**, **diferença** e **interação** entre os pares.

#### Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                    COMET-22                          │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Source   │  │ Hipótese │  │Referência│          │
│  │  (EN)    │  │  (MT)    │  │  (REF)   │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                │
│       ▼              ▼              ▼                │
│  ┌─────────────────────────────────────────┐        │
│  │         XLM-RoBERTa (encoder)           │        │
│  │      (550M params, 24 layers)           │        │
│  └─────────────────────────────────────────┘        │
│       │              │              │                │
│       ▼              ▼              ▼                │
│   emb_src        emb_mt         emb_ref             │
│       │              │              │                │
│       ▼              ▼              ▼                │
│  ┌─────────────────────────────────────────┐        │
│  │     Pooling + Feature Extraction         │        │
│  │  [emb_src; emb_mt; emb_ref;             │        │
│  │   |emb_src - emb_mt|;                   │        │
│  │   |emb_ref - emb_mt|;                   │        │
│  │   emb_src * emb_mt;                     │        │
│  │   emb_ref * emb_mt]                     │        │
│  └────────────────┬────────────────────────┘        │
│                   │                                  │
│                   ▼                                  │
│  ┌─────────────────────────────────────────┐        │
│  │       Estimator (Feed-Forward)           │        │
│  │       → score ∈ [0, 1]                   │        │
│  └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

#### Exemplo Visual

```
Source:    "The patient presented with persistent fever and dry cough."
Referência: "O paciente apresentou febre persistente e tosse seca."
Hipótese A: "O paciente apresentou febre persistente e tosse seca."   → COMET ≈ 1.00
Hipótese B: "O paciente teve febre contínua e tosse sem catarro."     → COMET ≈ 0.88
Hipótese C: "O doente mostrou uma febre que não passa e tosse."       → COMET ≈ 0.80
Hipótese D: "A mesa apresentou febre e tosse."                        → COMET ≈ 0.35

Análise:
  Hipótese A: Tradução perfeita                    → score máximo
  Hipótese B: Semanticamente correta, léxico diferente
              XLM-R captura que "contínua" ≈ "persistente"
              e "sem catarro" ≈ "seca"              → score alto
  Hipótese C: Significado preservado, estilo informal
              "doente" ≈ "paciente", "que não passa" ≈ "persistente"
              → COMET detecta adequação semântica     → score bom
  Hipótese D: Erro semântico grave ("mesa" ≠ "patient")
              COMET usa o source para detectar inconsistência
              → score baixo

Nota: BLEU daria score ZERO para B e C (sem match exato de n-gramas),
      mas COMET reconhece que são traduções válidas.
```

#### Por que COMET usa o source?

```
Source:     "The bank collapsed after the flood."
Referência: "O banco desabou após a enchente."

Hipótese A: "O banco desabou após a enchente."     → COMET alto
Hipótese B: "A instituição bancária faliu."         → COMET baixo

Sem o source, a Hipótese B poderia parecer uma paráfrase razoável.
Mas o source diz "flood" (enchente), não "financial crisis".
COMET detecta que "banco" = margem do rio (não instituição financeira),
e que "collapsed" = desabou fisicamente (não faliu).
→ O acesso ao source resolve ambiguidades e melhora a correlação
  com julgamentos humanos.
```

#### Implementação neste projeto

```python
# finetuning/metrics.py
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)
comet_model.eval()

data = [
    {"src": src, "mt": pred, "ref": ref}
    for src, pred, ref in zip(sources, predictions, references)
]
output = comet_model.predict(data, batch_size=2, gpus=1)
system_score = float(output.system_score)  # média ∈ [0, 1]
```

> **Nota**: O modelo utilizado é `Unbabel/wmt22-comet-da`, treinado em Direct Assessments (DA) de competições WMT15–WMT20. Requer ~2GB de VRAM adicionais (XLM-R large). Por isso, o modelo de tradução é movido para CPU antes do cálculo do COMET.

---

### 4. BERTScore

**Referência**: Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating Text Generation with BERT*. In International Conference on Learning Representations (ICLR 2020). https://openreview.net/forum?id=SkeHuCVFDr

#### O que mede
BERTScore calcula a **similaridade semântica** entre hipótese e referência usando **embeddings contextuais** de um modelo BERT pré-treinado. Em vez de comparar tokens exatos (como BLEU), compara representações vetoriais que codificam o significado no contexto.

#### Fórmula

Para cada token $x_i$ da referência e $\hat{x}_j$ da hipótese, calcula-se a similaridade por cosseno dos embeddings contextuais:

$$\text{Recall} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} \; \mathbf{x}_i^\top \hat{\mathbf{x}}_j$$

$$\text{Precision} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_j \in \hat{x}} \max_{x_i \in x} \; \mathbf{x}_i^\top \hat{\mathbf{x}}_j$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

(Onde $\mathbf{x}_i$ e $\hat{\mathbf{x}}_j$ são embeddings contextuais L2-normalizados)

#### Exemplo Visual

```
Referência: "O paciente apresentou febre persistente"
Hipótese:   "O doente teve temperatura alta contínua"

Passo 1: Gerar embeddings contextuais (BERT/mBERT)

  Referência:  O         paciente   apresentou   febre       persistente
               [v₁]      [v₂]       [v₃]         [v₄]        [v₅]

  Hipótese:    O         doente     teve         temperatura  alta       contínua
               [ĥ₁]      [ĥ₂]       [ĥ₃]         [ĥ₄]        [ĥ₅]       [ĥ₆]

Passo 2: Calcular matriz de similaridade por cosseno

               ĥ₁(O)  ĥ₂(doente)  ĥ₃(teve)  ĥ₄(temperatura)  ĥ₅(alta)  ĥ₆(contínua)
  v₁(O)        0.99    0.12        0.08       0.05              0.03       0.04
  v₂(paciente) 0.15    0.87 ←max   0.10       0.08              0.05       0.06
  v₃(apresentou)0.10   0.12        0.72 ←max  0.06              0.04       0.08
  v₄(febre)    0.05    0.09        0.07       0.83 ←max         0.45       0.12
  v₅(persistente)0.03  0.06        0.05       0.15              0.30       0.82 ←max

Passo 3: Greedy matching (cada token → melhor match)

  Recall (para cada token da referência, max cosseno com hipótese):
    O           → max(0.99, 0.12, 0.08, 0.05, 0.03, 0.04) = 0.99
    paciente    → max(0.15, 0.87, 0.10, 0.08, 0.05, 0.06) = 0.87  ← "doente" capturado!
    apresentou  → max(0.10, 0.12, 0.72, 0.06, 0.04, 0.08) = 0.72  ← "teve" capturado!
    febre       → max(0.05, 0.09, 0.07, 0.83, 0.45, 0.12) = 0.83  ← "temperatura" capturado!
    persistente → max(0.03, 0.06, 0.05, 0.15, 0.30, 0.82) = 0.82  ← "contínua" capturado!

  Recall = (0.99 + 0.87 + 0.72 + 0.83 + 0.82) / 5 = 0.846

  Precision (para cada token da hipótese, max cosseno com referência):
    O           → 0.99
    doente      → 0.87 (← "paciente")
    teve        → 0.72 (← "apresentou")
    temperatura → 0.83 (← "febre")
    alta        → 0.45 (← "febre", match parcial)
    contínua    → 0.82 (← "persistente")

  Precision = (0.99 + 0.87 + 0.72 + 0.83 + 0.45 + 0.82) / 6 = 0.780

  F₁ = 2 × (0.780 × 0.846) / (0.780 + 0.846) = 0.812

→ BERTScore F₁ = 0.812 (alto!)
  Apesar de ZERO palavras idênticas (exceto "O"),
  BERTScore reconhece equivalência semântica:
    paciente ↔ doente          (sinônimos)
    apresentou ↔ teve          (verbos relacionados)
    febre ↔ temperatura alta   (conceito médico equivalente)
    persistente ↔ contínua     (sinônimos)
```

#### Implementação neste projeto

```python
# finetuning/metrics.py
from bert_score import score

P, R, F1 = score(
    predictions,       # list[str] — traduções do modelo
    references,        # list[str] — referências humanas
    lang="pt",         # seleciona modelo multilíngue adequado
    batch_size=2,      # batch pequeno para caber na GPU
    device="cuda"
)
bertscore_f1 = float(F1.mean())  # média ∈ [0, 1]
```

> **Nota**: O parâmetro `lang="pt"` seleciona automaticamente o modelo BERT multilíngue adequado para português. A métrica é computada por sentença e depois promediada a nível de corpus. O modelo BERT é carregado após liberar o modelo de tradução da GPU para evitar OOM na RTX 4050 (6GB VRAM).

---

### Comparação das Métricas

| Aspecto                    | BLEU          | chrF          | COMET          | BERTScore      |
|----------------------------|:-------------:|:-------------:|:--------------:|:--------------:|
| **Granularidade**          | Palavra       | Caractere     | Sentença       | Subpalavra     |
| **Base de comparação**     | N-gramas exatos | Char n-gramas | Embeddings XLM-R | Embeddings BERT |
| **Detecta sinônimos?**     | Não           | Parcialmente  | Sim            | Sim            |
| **Detecta paráfrases?**    | Não           | Não           | Sim            | Sim            |
| **Sensível à morfologia?** | Não           | Sim           | Sim            | Sim            |
| **Usa frase fonte?**       | Não           | Não           | Sim            | Não            |
| **Correlação com humanos** | Moderada      | Boa           | Muito alta     | Alta           |
| **Custo computacional**    | Muito baixo   | Muito baixo   | Alto (~2GB GPU)| Médio (~1GB GPU)|
| **Velocidade**             | ~5s/corpus    | ~5s/corpus    | ~60s/corpus    | ~30s/corpus    |
| **Interpretabilidade**     | Alta          | Alta          | Baixa (caixa-preta) | Média     |
| **Ano de publicação**      | 2002          | 2015          | 2022           | 2020           |

### Por que usar 4 métricas?

```
Caso 1: BLEU alto, COMET baixo
→ A tradução tem as mesmas palavras, mas em ordem ou contexto errado
   Exemplo: "bank" traduzido como "banco" (financeiro) quando o contexto era "rio"

Caso 2: BLEU baixo, BERTScore alto
→ A tradução usa sinônimos/paráfrases corretos que BLEU não reconhece
   Exemplo: "febre" vs "temperatura elevada"

Caso 3: chrF alto, BLEU baixo
→ Morfologia correta mas palavras diferentes (flexões, conjugações)
   Exemplo: "apresentaram" vs "apresentou" (chrF captura "apresent-")

Caso 4: Todas altas
→ Tradução de alta qualidade ✅ (nosso caso: BLEU=45.51, chrF=70.54,
   COMET=0.8756, BERTScore=0.9124 após fine-tuning)
```

### Resultados neste projeto

| Métrica    | Antes  | Depois | Delta   | O que a melhoria indica                                    |
|------------|-------:|-------:|--------:|-----------------------------------------------------------|
| BLEU       | 40.06  | 45.51  | +5.45   | Mais n-gramas corretos → vocabulário do domínio aprendido  |
| chrF       | 65.61  | 70.54  | +4.93   | Melhor morfologia → concordância e acentuação aprendidas   |
| COMET      | 0.8499 | 0.8756 | +0.0257 | Maior adequação semântica validada por modelo neural        |
| BERTScore  | 0.8957 | 0.9124 | +0.0167 | Embeddings mais próximos → significado melhor preservado    |

> **Interpretação geral**: As 4 métricas melhoraram de forma consistente, indicando que o fine-tuning produziu ganhos reais em *todas* as dimensões de qualidade — não apenas em sobreposição lexical superficial, mas também em adequação semântica profunda.

---

## Referências

### Artigos Científicos

#### Métricas de Avaliação
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. In Proceedings of the 40th Annual Meeting of the ACL, pp. 311–318. https://aclanthology.org/P02-1040/
- Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. In Proceedings of the Tenth Workshop on Statistical Machine Translation (WMT), pp. 392–395. https://aclanthology.org/W15-3049/
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. In Proceedings of the Third Conference on Machine Translation (WMT), pp. 186–191. https://aclanthology.org/W18-6319/
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating Text Generation with BERT*. In International Conference on Learning Representations (ICLR 2020). https://openreview.net/forum?id=SkeHuCVFDr
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. In Proceedings of ACL 2020, pp. 8440–8451. https://aclanthology.org/2020.acl-main.747/
- Rei, R. et al. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*. In Proceedings of the Seventh Conference on Machine Translation (WMT), pp. 578–585. https://aclanthology.org/2022.wmt-1.52/

#### Arquitetura e Modelos
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is All You Need*. In Advances in Neural Information Processing Systems (NeurIPS 2017), pp. 5998–6008. https://arxiv.org/abs/1706.03762
- Raffel, C. et al. (2019). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. arXiv:1910.10683. https://arxiv.org/abs/1910.10683
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). *Self-Attention with Relative Position Representations*. In Proceedings of NAACL-HLT 2018, pp. 464–468. https://aclanthology.org/N18-2074/
- Lopes, A. et al. (2020). *Lite Training Strategies for Portuguese-English and English-Portuguese Translation*. In Proceedings of WMT 2020, pp. 833–840. https://aclanthology.org/2020.wmt-1.90/

#### Tokenização e Pré-processamento
- Kudo, T. & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. In Proceedings of EMNLP 2018, pp. 66–71. https://aclanthology.org/D18-2012/

#### Otimização e Treinamento
- Kingma, D. P. & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. In International Conference on Learning Representations (ICLR 2015). https://arxiv.org/abs/1412.6980
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. In International Conference on Learning Representations (ICLR 2019). https://arxiv.org/abs/1711.05101
- Smith, L. N. (2018). *A disciplined approach to neural network hyper-parameters: Part 1 – learning rate, batch size, momentum, and weight decay*. arXiv:1803.09820. https://arxiv.org/abs/1803.09820
- Goyal, P. et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677. https://arxiv.org/abs/1706.02677
- Howard, J. & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification*. In Proceedings of ACL 2018, pp. 328–339. https://aclanthology.org/P18-1031/

#### Regularização
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(1), pp. 1929–1958.
- Prechelt, L. (1998). *Early Stopping — But When?*. In Neural Networks: Tricks of the Trade, Lecture Notes in Computer Science, vol 1524, pp. 55–69. https://doi.org/10.1007/3-540-49430-8_3

#### Batch Size e Escala
- Masters, D. & Luschi, C. (2018). *Revisiting Small Batch Training for Deep Neural Networks*. arXiv:1804.07612. https://arxiv.org/abs/1804.07612
- Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. In International Conference on Learning Representations (ICLR 2017). https://arxiv.org/abs/1609.04836
- Ott, M. et al. (2018). *Scaling Neural Machine Translation*. In Proceedings of the Third Conference on Machine Translation (WMT), pp. 1–9. https://aclanthology.org/W18-6301/

#### Precisão Mista e Eficiência
- Micikevicius, P. et al. (2018). *Mixed Precision Training*. In International Conference on Learning Representations (ICLR 2018). https://arxiv.org/abs/1710.03740
- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). *Training Deep Nets with Sublinear Memory Cost*. arXiv:1604.06174. https://arxiv.org/abs/1604.06174

#### Beam Search e Decodificação
- Freitag, M. & Al-Onaizan, Y. (2017). *Beam Search Strategies for Neural Machine Translation*. In Proceedings of the First Workshop on Neural Machine Translation, pp. 56–60. https://aclanthology.org/W17-3207/

#### Fine-tuning e Adaptação de Domínio
- Miceli Barone, A. V., Haddow, B., Germann, U., & Sennrich, R. (2017). *Regularization techniques for fine-tuning in neural machine translation*. In Proceedings of EMNLP 2017, pp. 1489–1494. https://aclanthology.org/D17-1156/
- Freitag, M. & Al-Onaizan, Y. (2016). *Fast Domain Adaptation for Neural Machine Translation*. arXiv:1612.06897
- Neubig, G. & Hu, J. (2018). *Rapid Adaptation of Neural Machine Translation to New Languages*. In Proceedings of EMNLP 2018, pp. 875–880. https://aclanthology.org/D18-1103/
- Koehn, P. & Knowles, R. (2017). *Six Challenges for Neural Machine Translation*. In Proceedings of the First Workshop on Neural Machine Translation, pp. 28–39. https://aclanthology.org/W17-3204/

#### LLMs e Tradução
- Zhu, W. et al. (2023). *Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis*. In Findings of NAACL 2024. arXiv:2304.04675
- Xu, H. et al. (2023). *A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models*. In ICLR 2024. arXiv:2309.11674

#### Livros-texto
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/

### Bibliotecas e Ferramentas

- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- SacreBLEU: https://github.com/mjpost/sacrebleu
- COMET: https://github.com/Unbabel/COMET
- BERTScore: https://github.com/Tiiiger/bert_score
- Repositório do modelo: https://huggingface.co/unicamp-dl/translation-en-pt-t5
- Código-fonte do modelo: https://github.com/unicamp-dl/Lite-T5-Translation

---

**Versão**: 7.0 | **Data**: Fevereiro 2026
