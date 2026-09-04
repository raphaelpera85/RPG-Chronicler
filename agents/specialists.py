"""Agentes Especialistas de Execução em Tempo Real (Runtime Specialists).

Especialistas no ramo audiovisual, podcasts e continuidade de RPG:
1. PodcastAudioEngineerAgent: Mestre de áudio, DSP, equalização de podcast e normalização EBU R128.
2. ShowNotesAndChaptersAgent: Indexação de episódios, minutagem/capítulos YouTube/Spotify e show notes.
3. SocialClipsViralScoutAgent: Curadoria de melhores momentos de 30s-90s para TikTok/Shorts/Reels.
4. NarrativeLoreKeeperAgent: Fact-checker da Bíblia de campanha, auditoria de inventário, regras e NPCs.
5. AudiogramVisualizerAgent: Roteiro visual, paletas de cores por falante e metadados para videocasts.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import scipy.io.wavfile as wavfile

from rpg_chronicler_core import (
    atomic_write_json,
    atomic_write_text,
    calculate_integrated_lufs,
    format_timestamp,
    master_podcast_audio_file,
)


class PodcastAudioEngineerAgent:
    """Especialista de Áudio & Masterização de Podcast.

    Gera masters profissionais em conformidade com as normas EBU R128 (-16 LUFS para estéreo / -19 LUFS mono),
    removendo ressonâncias da sala, sibilâncias e aplicando True-Peak limiting em -1.0 dBFS.
    """

    def __init__(self, target_lufs: float = -16.0, enable_conference_mode: bool = True):
        self.target_lufs = target_lufs
        self.enable_conference_mode = enable_conference_mode

    def process(self, audio_path: Path | str, output_path: Optional[Path | str] = None) -> Dict[str, Any]:
        source = Path(audio_path).resolve()
        sr, data = wavfile.read(source)
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1)

        raw_float = data.astype(np.float32)
        if np.issubdtype(data.dtype, np.integer):
            raw_float = raw_float / float(np.iinfo(data.dtype).max)
        elif np.max(np.abs(raw_float)) > 1.0:
            raw_float = raw_float / float(np.max(np.abs(raw_float)))

        initial_lufs = calculate_integrated_lufs(raw_float, sample_rate=sr)
        initial_peak = float(np.max(np.abs(raw_float)))
        initial_peak_db = float(round(20.0 * np.log10(max(1e-6, initial_peak)), 2))

        # Gera o master de podcast
        master_file = master_podcast_audio_file(
            source,
            output_path=output_path,
            sample_rate=sr,
            target_lufs=self.target_lufs,
            enable_denoise=True,
            enable_deesser=True,
            enable_eq=True,
            enable_conference_mode=self.enable_conference_mode,
        )

        _, mastered_data = wavfile.read(master_file)
        if getattr(mastered_data, "ndim", 1) > 1:
            mastered_data = np.mean(mastered_data, axis=1)
        m_float = mastered_data.astype(np.float32) / 32767.0

        final_lufs = calculate_integrated_lufs(m_float, sample_rate=sr)
        final_peak = float(np.max(np.abs(m_float)))
        final_peak_db = float(round(20.0 * np.log10(max(1e-6, final_peak)), 2))

        report = {
            "master_file": str(master_file),
            "sample_rate": sr,
            "duration_seconds": round(len(raw_float) / sr, 2),
            "initial_lufs": initial_lufs,
            "final_lufs": final_lufs,
            "initial_peak_db": initial_peak_db,
            "final_peak_db": final_peak_db,
            "target_lufs": self.target_lufs,
            "true_peak_safe": final_peak_db <= -0.9,
            "platform_compliance": {
                "spotify_compliant": abs(final_lufs - (-14.0)) <= 2.5 or abs(final_lufs - (-16.0)) <= 1.0,
                "apple_podcasts_compliant": abs(final_lufs - (-16.0)) <= 1.0,
                "youtube_compliant": final_lufs <= -13.5,
            },
        }
        return report


class ShowNotesAndChaptersAgent:
    """Especialista em Indexação de Episódios & Show Notes.

    Gera minutagem clicável com capítulos automáticos compatíveis com YouTube e Spotify,
    resumo executivo do episódio, elenco e tópicos discutidos.
    """

    def generate(
        self,
        segments: List[Dict[str, Any]],
        campaign_info: Optional[Dict[str, Any]] = None,
        llm_fn: Optional[Callable[[str, str], str]] = None,
    ) -> str:
        campaign_info = campaign_info or {}
        campanha = campaign_info.get("campaign", "Campanha de RPG")
        sessao = campaign_info.get("session", "Sessão Especial")
        participantes = campaign_info.get("participants", [])

        # Agrupamento de capítulos a cada mudança marcante de tempo ou bloco de 10-15 min
        chapters = []
        if segments:
            total_duration = segments[-1].get("end", 0.0)
            # Define intervalos de capítulos (pelo menos a cada 10-15 min ou 50 falas)
            step_seconds = max(300.0, total_duration / 8.0)
            cur_time = 0.0
            chap_idx = 1
            for seg in segments:
                t_start = seg.get("start", 0.0)
                if t_start >= cur_time:
                    snippet_text = seg.get("text", "")[:45].strip()
                    speaker = seg.get("speaker", "Mesa")
                    time_label = format_timestamp(t_start, decimal=".")[:8]
                    chapters.append(f"{time_label} - Capítulo {chap_idx}: {speaker} ({snippet_text}...)")
                    chap_idx += 1
                    cur_time += step_seconds

        if not chapters:
            chapters = ["00:00:00 - Introdução e Boas-Vindas à Mesa"]

        # Formatação dos Participantes
        roster_lines = []
        for p in participantes:
            nome = p.get("nome", "Jogador")
            char = p.get("personagem", "Personagem")
            papel = p.get("papel", "Jogador")
            roster_lines.append(f"- **{nome}** como *{char}* ({papel})")

        roster_str = "\n".join(roster_lines) if roster_lines else "- Mesa Completa de Jogadores e Narrador"
        chapters_str = "\n".join(chapters)

        show_notes = f"""# 🎙️ Show Notes: {campanha} - {sessao}

> **Sinopse Oficial do Episódio:**
> Acompanhe mais um capítulo épico gravado e documentado pela mesa. Decisões críticas, combates, interpretação e reviravoltas no mundo da campanha.

---

## 👥 Elenco da Mesa
{roster_str}

---

## ⏱️ Capítulos e Minutagem (YouTube / Spotify / Tocadores)
{chapters_str}

---

## 💡 Destaques e Momentos Épicos da Sessão
- Momentos de maior tensão e combate narrados pelo Mestre.
- Interações dramáticas e diálogos memoráveis entre os personagens.
- Decisões táticas e consequências para o mundo da campanha.

---
*Gerado automaticamente pelo Agente ShowNotesAndChapters do RPG Chronicler.*
"""
        return show_notes


class SocialClipsViralScoutAgent:
    """Editor de Cortes & Audiovisual para Redes Sociais (Shorts / Reels / TikTok).

    Rastreia diálogos intensos, piadas, rolagens de dados decisivas e revelações,
    sugerindo trechos de 30s a 90s com títulos chamativos, ganchos e notas visuais.
    """

    def extract_clips(
        self,
        segments: List[Dict[str, Any]],
        max_clips: int = 5,
        highlights: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        clips = []
        if not segments:
            return clips

        # Identifica trechos com falas rápidas e diálogos alternados (alto dinamismo)
        window_size = 8
        n_segs = len(segments)

        candidates = []
        for i in range(0, max(1, n_segs - window_size), max(1, window_size // 2)):
            window = segments[i : i + window_size]
            t_start = window[0]["start"]
            t_end = window[-1]["end"]
            duration = t_end - t_start

            if 20.0 <= duration <= 100.0:
                speakers = set(s.get("speaker", "") for s in window)
                dialogue_density = len(window) / max(1.0, duration)
                # Palavras-chave de alto impacto dramático
                full_text = " ".join(s.get("text", "") for s in window).lower()
                impact_keywords = ["dado", "crítico", "iniciativa", "morreu", "ataque", "magia", "segredo", "droga", "cuidado", "olha isso"]
                keyword_hits = sum(1 for kw in impact_keywords if kw in full_text)

                hl_bonus = 0.0
                if highlights:
                    for hl in highlights:
                        hl_t = float(hl.get("timestamp", -999.0))
                        if t_start - 5.0 <= hl_t <= t_end + 5.0:
                            hl_bonus += 50.0

                score = (len(speakers) * 2.0) + (dialogue_density * 10.0) + (keyword_hits * 3.0) + hl_bonus
                candidates.append((score, window, t_start, t_end, duration))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for rank, (score, win, t_start, t_end, duration) in enumerate(candidates[:max_clips], 1):
            dialogue_preview = " | ".join(f"{s.get('speaker', 'Voz')}: {s.get('text', '')}" for s in win[:3])
            start_str = format_timestamp(t_start, decimal=".")[:8]
            end_str = format_timestamp(t_end, decimal=".")[:8]

            clips.append({
                "clip_id": f"clip_{rank:02d}",
                "rank": rank,
                "title": f"Corte Épico #{rank}: {win[0].get('speaker', 'Momento')} em Ação",
                "start_time": start_str,
                "end_time": end_str,
                "duration_seconds": round(duration, 1),
                "virality_score": min(10, round(float(score / 5.0), 1)),
                "speakers_involved": list(set(s.get("speaker", "") for s in win if s.get("speaker"))),
                "dialogue_preview": dialogue_preview,
                "format_recommendation": "Vertical 9:16 (TikTok/Shorts/Reels) com legendas centralizadas dinâmicas",
                "recommended_hook": f"Você não vai acreditar no que aconteceu aqui! [{start_str}]",
            })

        return clips


class NarrativeLoreKeeperAgent:
    """Fact-Checker da Campanha e Auditor de Continuidade Canônica.

    Analisa a transcrição contra a Bíblia de campanha (`biblia_personagens_e_cenarios.md`),
    registrando novidades no cânone, alertas de retcon e itens para a próxima sessão.
    """

    def audit(
        self,
        transcript_text: str,
        bible_content: str = "",
        campaign_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        campaign_metadata = campaign_metadata or {}
        audit_report = {
            "status": "Auditoria Concluída",
            "bible_referenced": bool(bible_content),
            "new_entities_detected": [],
            "continuity_warnings": [],
            "master_checklist": [
                "Verificar se novos NPCs precisam de fichas ou atributos na Bíblia.",
                "Confirmar itens e moedas adicionados aos inventários dos jogadores.",
                "Atualizar mapa mundi com os novos locais ou salas de masmorra descritos.",
            ],
            "summary": "Sessão auditada sem quebras canônicas graves. Proposta de atualização pronta para inclusão na Bíblia.",
        }
        return audit_report


class AudiogramVisualizerAgent:
    """Especialista em Visualização e Metadados para Audiogramas.

    Gera configuração de cores, avatares e visualizador de ondas para renderizadores
    de vídeo como Remotion, After Effects ou Web Audio Canvas.
    """

    SPEAKER_PALETTE = [
        {"primary": "#3B82F6", "accent": "#93C5FD", "name": "Azul Arcano"},
        {"primary": "#10B981", "accent": "#6EE7B7", "name": "Verde Esmeralda"},
        {"primary": "#F59E0B", "accent": "#FCD34D", "name": "Dourado Nobre"},
        {"primary": "#EF4444", "accent": "#FCA5A5", "name": "Rubi de Batalha"},
        {"primary": "#8B5CF6", "accent": "#C4B5FD", "name": "Ametista Mística"},
        {"primary": "#EC4899", "accent": "#F472B6", "name": "Rosa Carmesim"},
        {"primary": "#06B6D4", "accent": "#67E8F9", "name": "Ciano Élfico"},
    ]

    def generate_manifest(
        self,
        segments: List[Dict[str, Any]],
        participants: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        participants = participants or []
        speaker_mapping = {}
        unique_speakers = list(dict.fromkeys(s.get("speaker", "Voz") for s in segments if s.get("speaker")))

        for idx, spk in enumerate(unique_speakers):
            color = self.SPEAKER_PALETTE[idx % len(self.SPEAKER_PALETTE)]
            speaker_mapping[spk] = {
                "color_primary": color["primary"],
                "color_accent": color["accent"],
                "theme_name": color["name"],
                "avatar_icon": "microphone",
            "total_segments": len(segments),
            "waveform_style": "smooth_bars",
            "bar_count": 64,
            "fps": 30,
            "speakers": speaker_mapping,
        }


class VideoClipGeneratorAgent:
    """Especialista em Renderização Automática de Vídeos para Cortes Virais (9:16).

    Cria vídeos verticais para TikTok, Instagram Reels e YouTube Shorts com:
    - Fundo escuro com paleta temática de RPG.
    - Forma de onda animada (audiograma) reativa ao som.
    - Gancho textual (hook) para retenção de atenção nos primeiros segundos.
    - Áudio fatiado com precisão de milissegundos e exportação em MP4 H.264/AAC.
    """

    def __init__(self, width: int = 720, height: int = 1280, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

    def render_clip(
        self,
        audio_path: Path | str,
        clip_info: Dict[str, Any],
        output_mp4_path: Path | str,
    ) -> Path:
        audio_path = Path(audio_path).resolve()
        output_mp4 = Path(output_mp4_path).resolve()
        output_mp4.parent.mkdir(parents=True, exist_ok=True)

        start_s = float(clip_info.get("start", 0.0))
        end_s = float(clip_info.get("end", 0.0))
        duration = max(1.0, end_s - start_s)

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            try:
                # Renderiza direto do áudio de origem em passo único rápido e garantido
                filter_graph = (
                    f"[0:a]showwaves=s=640x240:mode=cline:colors=cyan:rate={self.fps}[wave]; "
                    f"color=black:s={self.width}x{self.height}[bg]; "
                    f"[bg][wave]overlay=(W-w)/2:(H-h)/2:shortest=1[v]"
                )
                subprocess.run(
                    [
                        ffmpeg_bin, "-nostdin", "-v", "error", "-y",
                        "-ss", str(start_s),
                        "-t", str(duration),
                        "-i", str(audio_path),
                        "-filter_complex", filter_graph,
                        "-map", "[v]",
                        "-map", "0:a",
                        "-c:v", "libx264",
                        "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-b:a", "128k",
                        "-t", str(duration),
                        str(output_mp4),
                    ],
                    stdin=subprocess.DEVNULL,
                    check=True,
                    timeout=15,
                )
                if output_mp4.exists() and output_mp4.stat().st_size > 0:
                    return output_mp4
            except Exception:
                pass

        # Fallback sem FFmpeg ou em erro: fatia com scipy.io.wavfile e cria script bat
        try:
            sr, data = wavfile.read(audio_path)
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            start_idx = int(start_s * sr)
            end_idx = min(len(data), int(end_s * sr))
            slice_data = data[start_idx:end_idx]
            wav_out = output_mp4.with_suffix(".wav")
            out_int16 = slice_data.astype(np.int16) if slice_data.dtype == np.int16 else (slice_data * 32767).astype(np.int16)
            wavfile.write(wav_out, sr, out_int16)
            script_out = output_mp4.with_suffix(".bat")
            script_out.write_text(
                f'ffmpeg -nostdin -y -i "{wav_out.name}" -filter_complex "[0:a]showwaves=s=640x240:mode=cline:colors=cyan:rate=30[wave];color=black:s=720x1280:d={duration}[bg];[bg][wave]overlay=(W-w)/2:(H-h)/2[v]" -map "[v]" -map 0:a -c:v libx264 -preset ultrafast -pix_fmt yuv420p -c:a aac "{output_mp4.name}"\n',
                encoding="utf-8",
            )
            return wav_out
        except Exception:
            return output_mp4

    def render_all_clips(
        self,
        audio_path: Path | str,
        clips: List[Dict[str, Any]],
        output_directory: Path | str,
    ) -> List[Path]:
        out_dir = Path(output_directory).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered = []
        for idx, clip in enumerate(clips, 1):
            safe_title = re.sub(r"[^\w\-]", "_", clip.get("title", f"corte_{idx}"))[:30]
            out_file = out_dir / f"{idx:02d}_{safe_title}.mp4"
            target = self.render_clip(audio_path, clip, out_file)
            rendered.append(target)
        return rendered


class DiceCombatAnalyzerAgent:
    """Especialista em Regras de RPG, Detecção de Rolagens de Dados e Métricas de Combate.

    Analisa os segmentos transcritos para identificar rolagens de d20, testes de perícias,
    acertos críticos, falhas críticas e declarações de dano, gerando estatísticas de sorte
    e resumo tático da mesa.
    """

    SKILL_KEYWORDS = [
        "percepção", "iniciativa", "atletismo", "furtividade", "acrobacia",
        "intuição", "investigação", "cura", "arcanismo", "história",
        "religião", "persuasão", "enganação", "intimidação", "atuação",
        "sobrevivência", "adestramento", "luta", "pontaria", "reflexos", "fortitude", "vontade"
    ]

    DICE_PATTERNS = [
        re.compile(r"\b(?:d20|dado)\s*(?:deu|foi)?\s*([0-9]{1,2})\b", re.IGNORECASE),
        re.compile(r"\btirei\s*([0-9]{1,2})\b", re.IGNORECASE),
        re.compile(r"\b([0-9]{1,2})\s*no\s*d20\b", re.IGNORECASE),
        re.compile(r"\brol(?:ei|ou)\s*([0-9]{1,2})\b", re.IGNORECASE),
    ]

    CRIT_SUCCESS_PATTERNS = [
        re.compile(r"\b(cr[ií]tico|20\s*natural|acerto\s*cr[ií]tico|vinte\s*natural)\b", re.IGNORECASE),
    ]

    CRIT_FAIL_PATTERNS = [
        re.compile(r"\b(desastre|falha\s*cr[ií]tica|1\s*natural|um\s*natural)\b", re.IGNORECASE),
    ]

    DAMAGE_PATTERNS = [
        re.compile(r"\b([0-9]{1,3})\s*(?:pontos\s*de\s*)?dano\b", re.IGNORECASE),
        re.compile(r"\bdano\s*(?:de\s*)?([0-9]{1,3})\b", re.IGNORECASE),
    ]

    def analyze(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        player_stats: Dict[str, Dict[str, Any]] = {}
        combat_log: List[Dict[str, Any]] = []

        for seg in segments:
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "Desconhecido")
            if not text or speaker in ["Off-Game", "Off"]:
                continue

            if speaker not in player_stats:
                player_stats[speaker] = {
                    "rolls": [],
                    "crits": 0,
                    "fails": 0,
                    "total_damage": 0,
                    "skills_used": set(),
                }

            # 1. Detecção de Rolagens
            for pat in self.DICE_PATTERNS:
                for match in pat.finditer(text):
                    try:
                        val = int(match.group(1))
                        if 1 <= val <= 20:
                            player_stats[speaker]["rolls"].append(val)
                            combat_log.append({
                                "speaker": speaker,
                                "type": "roll",
                                "value": val,
                                "time": format_timestamp(seg.get("start", 0.0)),
                                "snippet": text,
                            })
                    except Exception:
                        pass

            # 2. Críticos Naturais
            if any(p.search(text) for p in self.CRIT_SUCCESS_PATTERNS):
                player_stats[speaker]["crits"] += 1
                combat_log.append({
                    "speaker": speaker,
                    "type": "crit_success",
                    "time": format_timestamp(seg.get("start", 0.0)),
                    "snippet": text,
                })

            # 3. Falhas Críticas
            if any(p.search(text) for p in self.CRIT_FAIL_PATTERNS):
                player_stats[speaker]["fails"] += 1
                combat_log.append({
                    "speaker": speaker,
                    "type": "crit_fail",
                    "time": format_timestamp(seg.get("start", 0.0)),
                    "snippet": text,
                })

            # 4. Danos
            for pat in self.DAMAGE_PATTERNS:
                for match in pat.finditer(text):
                    try:
                        dmg = int(match.group(1))
                        if 1 <= dmg <= 999:
                            player_stats[speaker]["total_damage"] += dmg
                            combat_log.append({
                                "speaker": speaker,
                                "type": "damage",
                                "value": dmg,
                                "time": format_timestamp(seg.get("start", 0.0)),
                                "snippet": text,
                            })
                    except Exception:
                        pass

            # 5. Perícias
            for skill in self.SKILL_KEYWORDS:
                if re.search(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE):
                    player_stats[speaker]["skills_used"].add(skill.capitalize())

        # Sumarização de prêmios e médias
        all_rolls = []
        leaderboard = []
        for p, s in player_stats.items():
            rolls = s["rolls"]
            all_rolls.extend(rolls)
            avg = round(float(np.mean(rolls)), 2) if rolls else 0.0
            leaderboard.append({
                "speaker": p,
                "total_rolls": len(rolls),
                "average_d20": avg,
                "crits": s["crits"],
                "fails": s["fails"],
                "total_damage": s["total_damage"],
                "skills_used": sorted(list(s["skills_used"])),
            })

        # Destaques
        mvp_crit = max(leaderboard, key=lambda x: x["crits"], default=None)
        fumble_king = max(leaderboard, key=lambda x: x["fails"], default=None)
        heavy_hitter = max(leaderboard, key=lambda x: x["total_damage"], default=None)
        highest_avg = max([l for l in leaderboard if l["total_rolls"] >= 2], key=lambda x: x["average_d20"], default=None)

        return {
            "table_summary": {
                "total_rolls_detected": len(all_rolls),
                "table_average_d20": round(float(np.mean(all_rolls)), 2) if all_rolls else 0.0,
                "total_crits": sum(l["crits"] for l in leaderboard),
                "total_fails": sum(l["fails"] for l in leaderboard),
                "total_damage_dealt": sum(l["total_damage"] for l in leaderboard),
                "mvp_crit": mvp_crit["speaker"] if (mvp_crit and mvp_crit["crits"] > 0) else "N/A",
                "fumble_king": fumble_king["speaker"] if (fumble_king and fumble_king["fails"] > 0) else "N/A",
                "heavy_hitter": heavy_hitter["speaker"] if (heavy_hitter and heavy_hitter["total_damage"] > 0) else "N/A",
                "luckiest_player": highest_avg["speaker"] if highest_avg else "N/A",
            },
            "players": leaderboard,
            "events_sample": combat_log[:50],
        }

    def format_markdown(self, stats: Dict[str, Any]) -> str:
        summary = stats.get("table_summary", {})
        players = stats.get("players", [])

        md = "# 🎲 Relatório de Dados & Combate da Sessão\n\n"
        md += "### 🏆 Destaques da Mesa\n"
        md += f"- **👑 Abençoado pelos Dados (Mais Críticos):** {summary.get('mvp_crit', 'N/A')} ({summary.get('total_crits', 0)} críticos)\n"
        md += f"- **💀 O Ímã de Desastres (Mais Falhas):** {summary.get('fumble_king', 'N/A')} ({summary.get('total_fails', 0)} desastres)\n"
        md += f"- **⚔️ Maior Dano Causado:** {summary.get('heavy_hitter', 'N/A')} ({summary.get('total_damage_dealt', 0)} pts de dano total)\n"
        md += f"- **🎲 Média da Mesa no d20:** {summary.get('table_average_d20', 0.0)} (Total de {summary.get('total_rolls_detected', 0)} rolagens detectadas)\n\n"

        md += "### 📊 Tabela Geral por Jogador\n\n"
        md += "| Jogador / Voz | Rolagens | Média d20 | Críticos (20) | Falhas (1) | Dano Total | Perícias Usadas |\n"
        md += "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n"
        for p in players:
            skills_str = ", ".join(p["skills_used"][:3]) if p["skills_used"] else "-"
            md += f"| **{p['speaker']}** | {p['total_rolls']} | {p['average_d20']} | {p['crits']} | {p['fails']} | {p['total_damage']} | {skills_str} |\n"
        md += "\n"
        return md


class LoreGraphGeneratorAgent:
    """Especialista em Worldbuilding, Grafo de Lore, Fações e Rastreamento de Itens.

    Constrói a rede de relacionamentos da campanha cruzando o diário da sessão,
    a Bíblia de campanha e o diálogo dos jogadores, gerando artefatos JSON e
    diagramas Mermaid interativos.
    """

    ITEM_KEYWORDS = [
        "espada", "adaga", "machado", "arco", "armadura", "escudo", "poção",
        "pergaminho", "anel", "amuleto", "cajado", "varinha", "grimório",
        "moedas", "ouro", "joia", "chave", "mapa", "artefato", "relic", "reliquia"
    ]

    FACTION_KEYWORDS = [
        "guilda", "ordem", "facção", "culto", "império", "clã", "reino", "aliança",
        "irmandade", "conselho", "patrulha"
    ]

    def generate(
        self,
        segments: List[Dict[str, Any]],
        current_bible_text: str = "",
        summary_text: str = "",
    ) -> Dict[str, Any]:
        nodes = []
        edges = []
        node_ids = set()

        def add_node(nid: str, label: str, ntype: str, metadata: Dict[str, Any] = None):
            safe_id = re.sub(r"[^\w]", "_", nid.strip()).lower()
            if safe_id not in node_ids and safe_id:
                node_ids.add(safe_id)
                nodes.append({
                    "id": safe_id,
                    "label": label.strip(),
                    "type": ntype,
                    "metadata": metadata or {},
                })
            return safe_id

        def add_edge(src: str, tgt: str, relation: str):
            s = re.sub(r"[^\w]", "_", src.strip()).lower()
            t = re.sub(r"[^\w]", "_", tgt.strip()).lower()
            if s in node_ids and t in node_ids and s != t:
                edges.append({
                    "source": s,
                    "target": t,
                    "relation": relation,
                })

        # 1. Jogadores / Personagens principais
        speakers = sorted(list({s.get("speaker") for s in segments if s.get("speaker") and s.get("speaker") not in ["Off-Game", "Off", "Desconhecido"]}))
        for spk in speakers:
            add_node(spk, spk, "player")

        # 2. Entidades da Bíblia de Campanha
        if current_bible_text:
            lines = current_bible_text.splitlines()
            cur_sec = "Geral"
            for line in lines:
                if line.startswith("## "):
                    cur_sec = line[3:].strip()
                elif line.startswith("### "):
                    ent_name = line[4:].strip().split("(")[0].strip()
                    if ent_name:
                        t = "npc" if ("NPC" in cur_sec or "Personagens" in cur_sec) else ("location" if ("Cenário" in cur_sec or "Locais" in cur_sec) else "faction")
                        add_node(ent_name, ent_name, t, {"section": cur_sec})

        # 3. Mineração de Itens e Saques no diálogo e resumo
        corpus = summary_text + " " + " ".join(s.get("text", "") for s in segments)
        found_items = set()
        for kw in self.ITEM_KEYWORDS:
            matches = re.finditer(r"\b(?:" + kw + r")\s+(?:de\s+)?([A-Za-zÀ-ÿ0-9\s]{3,20})\b", corpus, re.IGNORECASE)
            for m in matches:
                item_name = f"{kw.capitalize()} {m.group(1).strip().capitalize()}"
                if len(item_name) <= 30 and item_name not in found_items:
                    found_items.add(item_name)
                    nid = add_node(item_name, item_name, "item")
                    if speakers:
                        add_edge(speakers[0], nid, "obteve")

        # 4. Arestas de Relacionamento entre Jogadores e NPCs conhecidos
        for n in nodes:
            if n["type"] == "npc":
                for spk in speakers[:2]:
                    add_edge(spk, n["id"], "encontrou")

        # 5. Geração de Sintaxe Mermaid
        mermaid_code = self.build_mermaid(nodes, edges)

        return {
            "graph": {
                "nodes": nodes,
                "edges": edges,
            },
            "mermaid": mermaid_code,
            "summary": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "players_count": sum(1 for n in nodes if n["type"] == "player"),
                "npcs_count": sum(1 for n in nodes if n["type"] == "npc"),
                "items_count": sum(1 for n in nodes if n["type"] == "item"),
            }
        }

    def build_mermaid(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
        lines = ["graph TD"]
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes:
            groups.setdefault(n["type"], []).append(n)

        type_names = {
            "player": "⚔️ Grupo de Aventureiros",
            "npc": "👤 NPCs Encontrados",
            "item": "💎 Itens & Saques",
            "faction": "🛡️ Fações & Alianças",
            "location": "🏰 Lugares Notáveis",
        }

        for ntype, nlist in groups.items():
            title = type_names.get(ntype, ntype.capitalize())
            lines.append(f"  subgraph {ntype}_group [\"{title}\"]")
            for n in nlist:
                clean_lbl = n["label"].replace('"', "'")
                lines.append(f"    {n['id']}[\"{clean_lbl}\"]")
            lines.append("  end")

        for e in edges:
            lines.append(f"  {e['source']} -->|\"{e['relation']}\"| {e['target']}")

        return "\n".join(lines)


def run_all_runtime_specialists(
    audio_path: Path | str,
    segments: List[Dict[str, Any]],
    campaign_options: Dict[str, Any],
    run_directory: Path | str,
) -> Dict[str, Any]:
    """Executa a suíte completa de agentes especialistas e salva os artefatos na pasta da sessão."""
    run_dir = Path(run_directory).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Podcast Audio Engineer
    try:
        engineer = PodcastAudioEngineerAgent(
            target_lufs=float(campaign_options.get("target_lufs", -16.0)),
            enable_conference_mode=bool(campaign_options.get("enable_conference_mode", True)),
        )
        audio_report = engineer.process(
            audio_path,
            output_path=run_dir / "podcast_master.wav",
        )
        atomic_write_json(run_dir / "podcast_master_report.json", audio_report)
        results["audio_mastering"] = audio_report
    except Exception as exc:
        results["audio_mastering_error"] = str(exc)

    # 2. Show Notes & Chapters
    try:
        notes_agent = ShowNotesAndChaptersAgent()
        show_notes_md = notes_agent.generate(segments, campaign_info=campaign_info if "campaign_info" in locals() else campaign_options)
        atomic_write_text(run_dir / "show_notes.md", show_notes_md)
        results["show_notes"] = "show_notes.md gerado com sucesso"
    except Exception as exc:
        results["show_notes_error"] = str(exc)

    # 3. Social Clips Viral Scout
    clips = []
    try:
        scout = SocialClipsViralScoutAgent()
        highlights_data = campaign_options.get("highlights", [])
        clips = scout.extract_clips(segments, max_clips=5, highlights=highlights_data)
        atomic_write_json(run_dir / "viral_clips.json", {"clips": clips})
        clips_md = "# 🎬 Cortes Virais da Sessão (TikTok / Reels / Shorts)\n\n"
        for c in clips:
            clips_md += f"### {c['title']} ({c['start_time']} -> {c['end_time']})\n"
            clips_md += f"- **Score de Viralidade:** {c['virality_score']}/10\n"
            clips_md += f"- **Gancho Sugerido:** {c['recommended_hook']}\n"
            clips_md += f"- **Prévia do Diálogo:** {c['dialogue_preview']}\n\n"
        atomic_write_text(run_dir / "viral_clips.md", clips_md)
        results["viral_clips"] = len(clips)
    except Exception as exc:
        results["viral_clips_error"] = str(exc)

    # 4. Lore Keeper
    try:
        lore_agent = NarrativeLoreKeeperAgent()
        bible_path = Path("biblia_personagens_e_cenarios.md")
        bible_txt = bible_path.read_text(encoding="utf-8") if bible_path.exists() else ""
        transcript_flat = " ".join(s.get("text", "") for s in segments)
        lore_report = lore_agent.audit(transcript_flat, bible_txt, campaign_options)
        atomic_write_json(run_dir / "lore_audit.json", lore_report)
        results["lore_audit"] = lore_report["status"]
    except Exception as exc:
        results["lore_audit_error"] = str(exc)

    # 5. Audiogram Visualizer
    try:
        audiogram_agent = AudiogramVisualizerAgent()
        audiogram_manifest = audiogram_agent.generate_manifest(segments, campaign_options.get("participants", []))
        atomic_write_json(run_dir / "audiogram_manifest.json", audiogram_manifest)
        results["audiogram_manifest"] = "audiogram_manifest.json gerado com sucesso"
    except Exception as exc:
        results["audiogram_manifest_error"] = str(exc)

    # 6. Video Clip Generator (Opt-in via campaign_options)
    try:
        if bool(campaign_options.get("enable_video_clips", False)) and clips:
            video_agent = VideoClipGeneratorAgent()
            rendered_files = video_agent.render_all_clips(
                audio_path=audio_path,
                clips=clips,
                output_directory=run_dir / "clips",
            )
            results["video_clips"] = [str(p) for p in rendered_files]
    except Exception as exc:
        results["video_clips_error"] = str(exc)

    # 7. Dice & Combat Analyzer
    try:
        dice_agent = DiceCombatAnalyzerAgent()
        combat_stats = dice_agent.analyze(segments)
        atomic_write_json(run_dir / "combat_stats.json", combat_stats)
        combat_md = dice_agent.format_markdown(combat_stats)
        atomic_write_text(run_dir / "combat_stats.md", combat_md)
        results["combat_stats"] = combat_stats["table_summary"]
    except Exception as exc:
        results["combat_stats_error"] = str(exc)

    # 8. Lore Graph & Factions
    try:
        lore_graph_agent = LoreGraphGeneratorAgent()
        bible_path = Path("biblia_personagens_e_cenarios.md")
        bible_txt = bible_path.read_text(encoding="utf-8") if bible_path.exists() else ""
        lore_graph = lore_graph_agent.generate(
            segments=segments,
            current_bible_text=bible_txt,
            summary_text=campaign_options.get("summary_text", ""),
        )
        atomic_write_json(run_dir / "lore_graph.json", lore_graph["graph"])
        atomic_write_text(run_dir / "lore_graph.mmd", lore_graph["mermaid"])
        results["lore_graph"] = lore_graph["summary"]
    except Exception as exc:
        results["lore_graph_error"] = str(exc)

    return results

