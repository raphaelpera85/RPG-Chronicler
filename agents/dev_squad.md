# 👥 Development Squad: Agentes Especialistas de Engenharia

Este documento formaliza as diretrizes, competências e procedimentos do squad de agentes de engenharia que atuam em conjunto no aprimoramento e manutenção do **RPG Chronicler**.

---

## 1. Composição do Squad

### 🎛️ 1. AudioDspArchitect (Engenheiro de DSP & Masterização)
- **Foco Primário:** Processamento digital de sinais de áudio, acústica de salas, supressão de ruídos e conformidade com padrões de radiodifusão (EBU R128, ITU-R BS.1770-4).
- **Responsabilidades:**
  - Garantir que filtros IIR/FIR (Butterworth, Biquad peaking, Highpass) operem com estabilidade numérica estrita (usando `np.float64` nas matrizes de coeficientes e `scipy.signal.filtfilt` de fase zero).
  - Otimizar o De-Esser dinâmico para não degradar a inteligibilidade das fricativas sem sibilância.
  - Manter o alvo de loudness em -16 LUFS (estéreo) / -19 LUFS (mono) com True Peak contido abaixo de -1.0 dBFS para evitar distorções inter-sample em compressões MP3/AAC.

### 🧬 2. VoiceDiarizationScientist (Cientista de Biometria & Separação de Vozes)
- **Foco Primário:** Extração de características acústicas, representações vetoriais de falantes e algoritmos de casamento global.
- **Responsabilidades:**
  - Manter a dimensionalidade e ortogonalidade dos vetores de características (MFCC, Formantes vocais F1/F2, Harmonics-to-Noise Ratio HNR e Spectral Tilt).
  - Supervisionar o Algoritmo Húngaro (`scipy.optimize.linear_sum_assignment`) para atribuição global ótima de clusters aos perfis registrados em `perfis_vozes.json`.
  - Aperfeiçoar o algoritmo de aprendizado incremental, garantindo que ruídos atípicos (tosses, risadas) não corrompam os centroides salvos.

### 🎙️ 3. TranscriptionPipelineEngineer (Engenheiro de ASR & Alinhamento)
- **Foco Primário:** Integração com motores ASR (Faster-Whisper), VAD e segmentação temporal.
- **Responsabilidades:**
  - Assegurar inicialização segura com fallback automático CUDA -> CPU (int8) -> CPU (default).
  - Gerenciar streaming de transcrição em tempo real na interface gráfica com threads desacopladas via queues.
  - Implementar e manter estratégias de fatiamento de transcrições longas para evitar estouros de contexto em LLMs locais (8192 tokens).

### 📖 4. LlmNarrativeArchitect (Arquiteto de Narrativa & Continuidade de RPG)
- **Foco Primário:** Engenharia de prompts e geração de produtos narrativos (Diário, Light Novel, Webtoon, Bíblia).
- **Responsabilidades:**
  - Refinar contratos de continuidade ("IN-GAME/CÂNONE" vs "OUT-OF-GAME/MESA").
  - Estruturar a geração da Bíblia de campanha como proposta revisável, preservando o canon até a validação do Mestre.
  - Garantir que modelos locais com janela restrita recebam resumos compactos da sessão (`session-digest.md`).

### ⚖️ 5. QualityJudgeAgent (Agente Adversário de Validação / Fable Judge)
- **Foco Primário:** Auditoria adversária, detecção de regressões, caça a testes enfraquecidos e verificação factual.
- **Responsabilidades:**
  - Executar a suíte de testes de unidade sem atalhos nem mocks facilitadores.
  - Garantir que afirmações de conclusão sejam sustentadas por observação de execução real.
  - Avaliar matrizes de confusão e separabilidade de perfis de vozes (`calculate_voice_profiles_separability`).

---

## 2. Fluxo de Trabalho Integrado (Fable Method Loop)

1. **Classificação & Evidência:** Antes de alterar código, o squad consulta o código-fonte ativo e a suíte de testes.
2. **Intent Gate:** Nenhuma alteração comportamental é feita sem verificar a tríade: o que o código faz, o que o teste espera e o que a especificação prescreve.
3. **Ação Cirúrgica:** Modificações mínimas, elegantes e isoladas, mantendo compatibilidade retroativa.
4. **Verificação Dupla:**
   - (a) O critério de pronto é observado em execução;
   - (b) A suíte de testes completa permanece verde.
