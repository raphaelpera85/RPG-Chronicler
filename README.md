# RPG Chronicler

Aplicativo local para gravar ou carregar sessões de RPG, transcrever em português, separar falantes por características acústicas e gerar diário, light novel, roteiro de webtoon e uma proposta revisável de Bíblia de continuidade.

## Recursos

- Gravação incremental WAV mono 16 kHz, sem manter a sessão inteira na memória.
- Importação de WAV, MP3, M4A, OGG, FLAC e AAC com conversão não destrutiva por FFmpeg.
- Faster-Whisper com tentativa de CUDA e fallback para CPU.
- Diarização acústica integrada à IA, calibração humana e perfis de voz persistentes.
- Aprendizado de perfis diretamente a partir da gravação da sessão, usando a quantidade de vozes configurada.
- LM Studio local ou APIs OpenAI-compatible.
- Checkpoint automático por execução em `runs/`.
- Transcrição em Markdown, JSON, SRT e VTT.
- Fatiamento automático da transcrição para modelos locais com janela curta, incluindo LM Studio em 8192 tokens.
- Cancelamento e bloqueio contra execuções concorrentes.
- Bíblia gerada como proposta: só substitui o canon após aprovação, mantendo backup.

## Requisitos

- Windows 10/11.
- Python 3.11 ou superior.
- FFmpeg e FFprobe no `PATH`.
- Microfone, para gravação ao vivo.
- Opcional: GPU NVIDIA/CUDA compatível com Faster-Whisper.
- Opcional: LM Studio ou uma chave de API OpenAI-compatible.

## Instalação no Windows

1. Instale Python e FFmpeg.
2. Execute `setup_windows.bat`.
3. Execute `iniciar_gravador_rpg.bat`.

Instalação manual:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe rpg_chronicler.py
```

## Configuração da IA

O modo recomendado para privacidade é LM Studio em `http://localhost:1234/v1`.

Para um provedor em nuvem, informe a chave apenas na interface ou na variável de ambiente:

```powershell
$env:RPG_CHRONICLER_API_KEY = "sua-chave"
```

A chave não é persistida por versões novas do aplicativo. Consulte [SECURITY.md](SECURITY.md) antes de usar gravações reais.

## Fluxo de uso

1. Configure campanha, sessão, participantes e vocabulário.
2. Na aba **LM Studio & Timbre**, defina **Quantidade de Vozes na Mesa**.
3. Grave ou carregue um áudio.
4. Inicie o processamento e, se habilitado, confirme as amostras de voz.
5. A IA recebe as predições acústicas, confirmações humanas e exemplos por voz para refinar a identificação.
6. Acompanhe os checkpoints em `runs/<run-id>/`.
7. Revise a proposta da Bíblia e clique em **Aprovar Bíblia** apenas quando estiver correta.
8. Use **Salvar Tudo** para uma exportação Markdown agregada.

### Treino pela gravação da sessão

Para uma sessão real, carregue a gravação, defina a quantidade de vozes esperada e deixe **Aprender/reforçar perfis de voz a partir da gravação da sessão** ligado. O Chronicler agrupa a sessão nesse número de vozes, extrai amostras, compara com `perfis_vozes.json` e atualiza o banco quando houver confirmação humana ou reconhecimento acústico confiante. Se o grupo tiver vozes novas, mantenha a confirmação interativa ligada na primeira sessão para nomear cada voz corretamente.

## Checkpoints

Cada execução pode conter:

- `manifest.json`: áudio, configuração sem segredos e estado das etapas.
- `transcript.json`, `transcript.md`, `transcript.srt`, `transcript.vtt`.
- `transcript-refined.md`, `diary.md`, `novel.md`, `webtoon.md`.
- `bible-proposal.md`.

Se a IA falhar, a transcrição e a diarização já concluídas permanecem no checkpoint. Os checkpoints são registros versionados para auditoria e recuperação manual; a retomada automática de uma etapa ainda não é oferecida.

## Privacidade e nuvem

Por padrão, o endpoint é local. Ao usar um endpoint remoto, o aplicativo pede consentimento para o destino e modelo específicos antes de testar a conexão ou enviar transcrições. Chaves devem ser fornecidas pela variável de ambiente `RPG_CHRONICLER_API_KEY`; não são gravadas no JSON.

### Usar a assinatura do Hermes

O Hermes possui um proxy local oficial que faz o login OAuth no navegador e encaminha as chamadas para a assinatura sem expor o token ao RPG Chronicler. Depois de instalar e autenticar o Hermes, execute `iniciar_com_hermes.bat`. Ele inicia `hermes proxy start` em uma janela separada e aponta o aplicativo para `http://127.0.0.1:8645/v1`. O Hermes deve permanecer aberto durante o processamento.

Esse fluxo usa o proxy local documentado pelo Hermes; não copie `auth.json`, refresh tokens ou cookies para este projeto.

Para ativar a diarização neural, instale o extra `requirements-ml.txt`, aceite o modelo no Hugging Face e defina `HF_TOKEN` antes de iniciar. O aplicativo usará automaticamente `pyannote/speaker-diarization-community-1` e continuará com o diarizador acústico se o extra não estiver disponível.

### Modelos locais com 8192 tokens

Na aba **LM Studio & Timbre**, mantenha **Janela de contexto** em `8192` para modelos locais comuns. O RPG Chronicler divide automaticamente transcrições longas em fatias menores, refina cada fatia separadamente e cria um `session-digest.md` compacto antes de gerar diário, light novel, webtoon e proposta de Bíblia. Se o modelo local ainda reclamar de contexto, reduza esse valor para `4096` ou `6144`.

## Testes

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Os testes usam diretórios temporários e não modificam gravações, configurações, perfis ou Bíblia reais.

## Dados privados

Os seguintes caminhos são deliberadamente ignorados pelo Git: configuração local, gravações, snippets, transcrições, runs, backups, perfis vocais e Bíblia real. O repositório remoto deve conter apenas código, documentação e exemplos sanitizados.

## Estado atual

Esta é a versão inicial `0.1.0`. O pipeline de unidade e integração local é testável sem rede; microfone real, GPU e cada provedor de nuvem continuam sendo integrações opcionais dependentes do ambiente.
