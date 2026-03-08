import asyncio
from pathlib import Path


async def run_ffmpeg(args: list[str]) -> None:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode().strip() or "FFmpeg command failed")


async def create_placeholder_video(output_path: Path, duration: int, text: str, ffmpeg_binary: str = "ffmpeg") -> None:
    safe_text = (text or "Manga scene").replace(":", " ").replace("'", "").replace("\n", " ")[:80]
    args = [
        ffmpeg_binary,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=1280x720:d={duration}",
        "-vf",
        f"drawtext=text='{safe_text}':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=(h-text_h)/2",
        str(output_path),
    ]
    await run_ffmpeg(args)


async def create_silent_audio(output_path: Path, duration: int, ffmpeg_binary: str = "ffmpeg") -> None:
    args = [
        ffmpeg_binary,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=44100:cl=stereo:d={duration}",
        "-q:a",
        "4",
        str(output_path),
    ]
    await run_ffmpeg(args)


async def write_srt_file(output_path: Path, text: str, duration: int) -> None:
    content = "1\n00:00:00,000 --> 00:00:%02d,000\n%s\n" % (max(duration, 1), text.strip() or "[instrumental]")
    await asyncio.to_thread(output_path.write_text, content, "utf-8")


async def mix_background_music(ffmpeg_binary: str, narration_path: Path, music_path: Path, output_path: Path) -> None:
    args = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(narration_path),
        "-stream_loop",
        "-1",
        "-i",
        str(music_path),
        "-filter_complex",
        "[1:a]volume=0.18[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map",
        "[aout]",
        str(output_path),
    ]
    await run_ffmpeg(args)


async def mux_audio_video(
    ffmpeg_binary: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    subtitles_path: Path | None = None,
) -> None:
    if subtitles_path:
        subtitle_source = subtitles_path.as_posix().replace(":", "\\:")
        args = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-vf",
            f"subtitles='{subtitle_source}'",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    else:
        args = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    await run_ffmpeg(args)
