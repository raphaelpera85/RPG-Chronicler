"""Agentes Especialistas do RPG Chronicler.

Pacote contendo o squad de desenvolvimento e os agentes de execução em tempo real
para produção audiovisual, podcast, masterização e continuidade narrativa.
"""

from .specialists import (
    PodcastAudioEngineerAgent,
    ShowNotesAndChaptersAgent,
    SocialClipsViralScoutAgent,
    NarrativeLoreKeeperAgent,
    AudiogramVisualizerAgent,
    VideoClipGeneratorAgent,
    DiceCombatAnalyzerAgent,
    LoreGraphGeneratorAgent,
    run_all_runtime_specialists,
)

__all__ = [
    "PodcastAudioEngineerAgent",
    "ShowNotesAndChaptersAgent",
    "SocialClipsViralScoutAgent",
    "NarrativeLoreKeeperAgent",
    "AudiogramVisualizerAgent",
    "VideoClipGeneratorAgent",
    "DiceCombatAnalyzerAgent",
    "LoreGraphGeneratorAgent",
    "run_all_runtime_specialists",
]

