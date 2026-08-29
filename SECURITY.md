# Segurança e privacidade

- Nunca faça commit de `rpg_chronicler_config.json`, `.env`, gravações, transcrições, snippets ou perfis vocais.
- Informe a chave do provedor apenas durante a sessão ou pela variável `RPG_CHRONICLER_API_KEY`.
- Tokens de ChatGPT/Codex/OpenCode não são chaves de API e não são importados pelo aplicativo.
- Ao usar um provedor em nuvem, a transcrição e a Bíblia são enviadas ao endpoint configurado. Use LM Studio para processamento local.
- Perfis vocais e snippets podem ser considerados dados biométricos. Obtenha consentimento dos participantes e defina uma política de retenção.
- Se uma chave já foi salva em texto claro, revogue-a no provedor antes de apagar o arquivo local.

Falhas de segurança não devem incluir credenciais, gravações ou transcrições reais em relatórios públicos.

