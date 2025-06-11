# bot_music_bot.py – /icecream  +  /instrument  +  /genre
import os, io, sys, subprocess, csv, pathlib, asyncio, typing as t

import numpy as np
import requests, ffmpeg, discord, torchaudio, torch
from discord.ext import commands
from discord import app_commands, ui, ButtonStyle
from dotenv import load_dotenv

# ─── 0. Load ENV tokens ─────────────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
AUDD_TOKEN    = os.getenv("AUDD_TOKEN")
if not (DISCORD_TOKEN and AUDD_TOKEN):
    raise RuntimeError(".env thiếu DISCORD_TOKEN hoặc AUDD_TOKEN")

# ─── 1. PANNs-inference ─────────────────────────────────────────────────────
try:
    from panns_inference import AudioTagging
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "panns_inference"])
    from panns_inference import AudioTagging

device = "cuda" if torch.cuda.is_available() else "cpu"
tagger = AudioTagging(device=device)

# ─── 2. AudioSet label table ─────────────────────────────────────────────────
CSV_PATH = pathlib.Path.home() / "panns_data" / "class_labels_indices.csv"
if not CSV_PATH.exists():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/qiuqiangkong/panns_inference/main/"
        "panns_inference/data/class_labels_indices.csv",
        CSV_PATH
    )
LABELS_FULL = {
    int(r["index"]): r["display_name"]
    for r in csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8"))
}

# ─── 3. Whitelist nhạc cụ & thể loại ────────────────────────────────────────
INSTRUMENT_WORDS = (
    "guitar","banjo","mandolin","ukulele","bass","double bass","electric bass",
    "piano","keyboard","organ","accordion","harpsichord","violin","fiddle",
    "viola","cello","saxophone","trumpet","trombone","tuba","clarinet","flute",
    "drum","snare","cymbal","hi-hat","ride cymbal","tom-tom","percussion",
    "marimba","xylophone","vibraphone","harmonica","bagpipe","harp"
)
GENRE_MAP = {
    "rock": ["rock","hard rock","punk rock"],
    "pop": ["pop","dance pop","synthpop"],
    "hip hop": ["hip hop","rap"],
    "jazz": ["jazz","smooth jazz","bebop"],
    "blues": ["blues"],
    "classical": ["classical","orchestra","symphony"],
    "electronic": ["electronic","edm"],
    "techno": ["techno"],
    "house": ["house"],
    "trance": ["trance"],
    "dubstep": ["dubstep"],
    "metal": ["metal","death metal","black metal"],
    "punk": ["punk"],
    "country": ["country"],
    "bluegrass": ["bluegrass"],
    "folk music": ["folk","folk music"],
    "reggae": ["reggae"],
    "salsa": ["salsa"],
    "latin": ["latin"],
    "samba": ["samba"],
}
def build_id2genre(labels: dict[int,str]) -> dict[int,str]:
    out: dict[int,str] = {}
    for idx, name in labels.items():
        low = name.lower()
        for genre, kws in GENRE_MAP.items():
            if any(kw in low for kw in kws):
                out[idx] = genre
                break
    return out

ID2INST  = {i:n for i,n in LABELS_FULL.items() if any(k in n.lower() for k in INSTRUMENT_WORDS)}
ID2GENRE = build_id2genre(LABELS_FULL)

# ─── 4. Common helpers ──────────────────────────────────────────────────────
def pan_topk(wav: torch.Tensor, sr: int, id2name: dict[int,str], k: int = 3):
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.dim() == 2:
        wav = wav.mean(0, keepdim=True)
    scores = tagger.inference(np.expand_dims(wav.squeeze(0).numpy(), 0))
    clip   = scores[0] if isinstance(scores, tuple) else scores["clipwise_output"]
    clip   = clip[0] if clip.ndim == 2 else clip
    idx    = [i for i in np.argsort(-clip) if i in id2name][:k]
    return [(id2name[i], float(clip[i])) for i in idx]

def pan_aggregate(wav: torch.Tensor, sr: int, id2name: dict[int,str],
                  k: int = 5, window_s: float = 5.0, hop_s: float = 2.5):
    if wav.dim() == 2: wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    total, win, hop = wav.size(1), int(window_s*sr), int(hop_s*sr)
    clips = []
    for st in range(0, max(1, total-win+1), hop):
        seg = wav[:, st:st+win]
        scores = tagger.inference(np.expand_dims(seg.mean(0).numpy(),0))
        out    = scores[0] if isinstance(scores, tuple) else scores["clipwise_output"]
        arr    = out[0] if out.ndim == 2 else out
        clips.append(torch.tensor(arr))
    if not clips: return []
    avg = torch.stack(clips,0).mean(0).numpy()
    idx = [i for i in np.argsort(-avg) if i in id2name][:k]
    return [(id2name[i], float(avg[i])) for i in idx]

def to_mp3(raw: bytes) -> bytes:
    out,_ = ffmpeg.run(
        ffmpeg.input("pipe:0").output("pipe:1",
            format="mp3", ac=1, ar="44100", audio_bitrate="128k"),
        input=raw, capture_stdout=True, capture_stderr=True, overwrite_output=True)
    return out

def audd_identify(mp3: bytes) -> t.Optional[dict]:
    r = requests.post("https://api.audd.io/",
        data={"api_token":AUDD_TOKEN,"return":"spotify"},
        files={"file":("clip.mp3",mp3)},timeout=20).json()
    return r.get("result") if r.get("status")=="success" else None

def assess_quality(wav: torch.Tensor, sr: int) -> str:
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.dim() == 2:
        wav = wav.mean(0,keepdim=True)
    rms = wav.pow(2).mean().sqrt().item()
    if sr>=32000 and rms>0.03: return "Tốt"
    if rms<0.01 or sr<16000:   return "Kém"
    return "Trung bình"

# ─── 5. Discord bot setup ───────────────────────────────────────────────────
bot = commands.Bot(command_prefix=commands.when_mentioned,
                   intents=discord.Intents.default())

# /icecream -----------------------------------------------------------------
@bot.tree.command(name="icecream", description="Identify the song in an audio clip")
@app_commands.describe(audio="Upload an audio file (≥6 s)")
async def icecream(inter: discord.Interaction, audio: discord.Attachment):
    # 1) lấy tên file + định dạng
    fname = audio.filename
    fmt   = pathlib.Path(fname).suffix.lstrip(".").lower()

    # 2) kiểm tra file
    if not (audio.content_type or "").startswith("audio/"):
        return await inter.response.send_message("❌ File không phải audio.", ephemeral=True)

    # 3) defer vì sẽ chạy FFmpeg + API
    await inter.response.defer()

    # 4) convert + gọi AudD
    raw = await audio.read()
    try:
        mp3 = await asyncio.to_thread(to_mp3, raw)
    except ffmpeg.Error:
        return await inter.followup.send("❌ Không đọc được file âm thanh.")
    song = audd_identify(mp3)
    if not song:
        return await inter.followup.send("⚠️ Chưa nhận diện được bài.")

    # 5) build embed
    title, artist = song.get("title"), song.get("artist")
    link = (song.get("spotify") or {}).get("external_urls",{}).get("spotify") or song.get("song_link")
    embed = discord.Embed(title=f"{title} – {artist}", url=link, colour=0x1DB954)\
            .set_author(name="Spotify",
                        icon_url="https://raw.githubusercontent.com/encharm/Font-Awesome-SVG-PNG/master/black/png/64/spotify.png")
    # thêm field File...
    embed.add_field(name="File", value=f"{fname} (`.{fmt}`)", inline=False)

    album = song.get("album") if isinstance(song.get("album"),dict) else {}
    embed.add_field(name="Album",value=album.get("title","—"),inline=True)
    embed.add_field(name="Year", value=(song.get("release_date","—")[:4] or "—"),inline=True)
    dur = song.get("duration") or song.get("duration_ms",0)/1000
    mm,ss = divmod(int(dur),60)
    embed.add_field(name="Duration",value=(f"{mm}:{ss:02d}" if dur else "—"),inline=True)

    cover = next((album.get(k) for k in ("cover","cover_big","cover_medium","image") if album.get(k)),None)
    if not cover:
        imgs = (song.get("spotify") or {}).get("album",{}).get("images",[])
        if imgs: cover = imgs[0].get("url")
    if cover: embed.set_image(url=cover)

    preview = (song.get("spotify") or {}).get("preview_url")
    file_obj = None
    if preview:
        data = requests.get(preview,timeout=15).content
        file_obj = discord.File(io.BytesIO(data),"preview.mp3")
        embed.set_footer(text="🎧 30-second preview attached")

    view = ui.View()
    view.add_item(ui.Button(label="Play on Spotify", style=ButtonStyle.link, url=link))

    # 6) gửi followup
    await inter.followup.send(embed=embed, file=file_obj, view=view)

# /instrument ---------------------------------------------------------------
@bot.tree.command(name="instrument", description="Detect instruments in an audio clip")
@app_commands.describe(audio="Upload an audio file (≥3 s)")
async def instrument(inter: discord.Interaction, audio: discord.Attachment):
    fname = audio.filename
    fmt   = pathlib.Path(fname).suffix.lstrip(".").lower()

    if not (audio.content_type or "").startswith("audio/"):
        return await inter.response.send_message("❌ File không phải audio.", ephemeral=True)
    await inter.response.defer()

    raw = await audio.read()
    wav, sr = torchaudio.load(io.BytesIO(raw))

    quality = assess_quality(wav, sr)
    instrs  = await asyncio.to_thread(pan_topk, wav, sr, ID2INST, 3)
    if not instrs:
        inst_line = "🤷 Bot không chắc nhạc cụ nào."
    else:
        inst_line = " | ".join(f"Top{i+1}: {n} ({p*100:.0f}%)" for i,(n,p) in enumerate(instrs))

    reply = (
        f"File: `{fname}` (`.{fmt}`)\n"
        f"Chất lượng audio: **{quality}**\n"
        f"Nhạc cụ khả thi:\n{inst_line}"
    )
    await inter.followup.send(reply)

# /genre --------------------------------------------------------------------
@bot.tree.command(name="genre", description="Detect musical genres in an audio clip")
@app_commands.describe(audio="Upload an audio file (≥3 s)")
async def genre(inter: discord.Interaction, audio: discord.Attachment):
    fname = audio.filename
    fmt   = pathlib.Path(fname).suffix.lstrip(".").lower()

    if not (audio.content_type or "").startswith("audio/"):
        return await inter.response.send_message("❌ File không phải audio.", ephemeral=True)
    await inter.response.defer()

    raw = await audio.read()
    wav, sr = torchaudio.load(io.BytesIO(raw))
    if wav.dim()==2: wav = wav.mean(0,keepdim=True)

    genres = await asyncio.to_thread(pan_aggregate, wav, sr, ID2GENRE, 5)
    if not genres:
        genres = await asyncio.to_thread(pan_topk, wav, sr, ID2GENRE, 3)
    genres = sorted(genres, key=lambda x:-x[1])[:3]
    if not genres:
        return await inter.followup.send("🤷 Bot không chắc thể loại nào.")

    lines = "\n".join(f"• {n} ({p*100:.1f}%)" for n,p in genres)
    reply = (
        f"File: `{fname}` (`.{fmt}`)\n"
        f"🎼 **Thể loại khả thi (Top 3):**\n{lines}"
    )
    await inter.followup.send(reply)

# ready ----------------------------------------------------------------------
@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user} – all commands ready!")

bot.run(DISCORD_TOKEN)
