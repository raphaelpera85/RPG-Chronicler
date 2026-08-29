# Mapas em português (uso pessoal)

`traduzir_mapa_ptbr.py` gera uma cópia do PDF do Inner Sea com OCR e
sobreposição de alguns rótulos geográficos traduzíveis. O arquivo original
nunca é alterado.

## Executar

1. Instale as dependências: `..\\.venv\\Scripts\\python.exe -m pip install -r requirements-map.txt`.
2. Confirme que o Tesseract está instalado com o idioma `por`.
3. Execute `traduzir_mapa_ptbr.bat`.

Saída: `D:\\Users\\rapha\\Documents\\Projetos\\RPG\\livros\\inner-sea-poster-map-folio-ptbr.pdf`.

## Limitações

Esta é uma tradução assistida por OCR, não uma localização editorial completa:
nomes próprios de regiões, cidades e personagens são preservados; alguns
rótulos decorativos ou textos pequenos podem permanecer em inglês; e as áreas
recobertas podem ter uma tonalidade diferente do fundo cartográfico. Revise as
seis páginas antes de imprimir. Para uma versão realmente integral, é preciso
revisão manual página a página ou uma fonte vetorial licenciada.
