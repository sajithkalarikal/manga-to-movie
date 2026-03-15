import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def run_ffmpeg(args: list[str]) -> None:
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        binary = args[0] if args else "ffmpeg"
        raise RuntimeError(
            f"FFmpeg binary not found: '{binary}'. Install ffmpeg and/or set FFMPEG_BINARY to its absolute path."
        ) from exc
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode().strip() or "FFmpeg command failed")


async def create_placeholder_video(
    output_path: Path,
    duration: int,
    text: str,
    ffmpeg_binary: str = "ffmpeg",
    image_path: Path | None = None,
) -> None:
    if image_path is not None and image_path.exists():
        zoom_frames = max(duration * 25, 25)
        image_args = [
            ffmpeg_binary,
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-vf",
            (
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,"
                f"zoompan=z='min(zoom+0.0008,1.08)':d={zoom_frames}:s=1280x720"
            ),
            "-t",
            str(duration),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            await run_ffmpeg(image_args)
            return
        except RuntimeError as exc:
            logger.warning("Image-based placeholder video generation failed; falling back to color background. error=%s", exc)

    safe_text = (text or "Manga scene").replace(":", " ").replace("'", "").replace("\n", " ")[:80]
    args_with_text = [
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
    try:
        await run_ffmpeg(args_with_text)
        return
    except RuntimeError as exc:
        error_message = str(exc)
        if "No such filter: 'drawtext'" not in error_message and "Filter not found" not in error_message:
            raise
        logger.warning("FFmpeg drawtext filter unavailable; creating placeholder video without text.")

    args_without_text = [
        ffmpeg_binary,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=1280x720:d={duration}",
        str(output_path),
    ]
    await run_ffmpeg(args_without_text)


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
        subtitle_source = subtitles_path.as_posix()
        # Escape characters that have meaning inside FFmpeg filtergraphs.
        subtitle_source = (
            subtitle_source.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace(",", "\\,")
        )
        args = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-vf",
            f"subtitles=filename={subtitle_source}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        try:
            await run_ffmpeg(args)
            return
        except RuntimeError as exc:
            msg = str(exc)
            if "No such filter: 'subtitles'" not in msg and "Filter not found" not in msg:
                raise
            logger.warning("FFmpeg subtitles filter unavailable; muxing without burned-in subtitles.")
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
