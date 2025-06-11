Discord Bot: 

#1 clone code về
git clone:

#2 tạo virtual enviroment
--- run in Powershell ---

$ python -m venv .venv
$ source venv/bin/activate #linux

.\.venv\Scripts\Activate.ps1 #PS Window

if error chạy lệnh này:" Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass " trước và thử lại


#3 cai dependencies python
$ pip install --upgrade pip
$ python -m pip install -U discord.py[voice] requests ffmpeg-python python-dotenv
$ pip install torch torchaudio
$ pip install panns-inference 

#4 download ffmpeg 
https://ffmpeg.org/download.html
add C:\ffmpeg\bin to PATH (enviroment system variable)

#run Bot
$ python bot_id_song.py