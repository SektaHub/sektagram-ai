from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cpu", compute_type="int8")


language = "auto"
initial_prompt = "Please do not translate, only transcription be allowed."
word_level_timestamps = False
vad_filter = True
vad_filter_min_silence_duration_ms = 50
text_only = True

def seconds_to_time_format(s):
    # Convert seconds to hours, minutes, seconds, and milliseconds
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    seconds = s // 1
    milliseconds = round((s % 1) * 1000)

    # Return the formatted string
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def transcribe_audio_file(file_path):
    # Transcribe audio
    segments, info = model.transcribe(file_path, beam_size=5,
                                      language=None if language == "auto" else language,
                                      initial_prompt=initial_prompt,
                                      word_timestamps=word_level_timestamps,
                                      vad_filter=vad_filter,
                                      vad_parameters=dict(min_silence_duration_ms=vad_filter_min_silence_duration_ms))

    transcription = ""
    sentence_idx = 1
    for segment in segments:
        if word_level_timestamps:
            for word in segment.words:
                ts_start = seconds_to_time_format(word.start)
                ts_end = seconds_to_time_format(word.end)
                if text_only:
                    transcription += f"{word.word} "
                else:
                    transcription += f"[{ts_start} --> {ts_end}] {word.word}\n"
                    sentence_idx += 1
        else:
            ts_start = seconds_to_time_format(segment.start)
            ts_end = seconds_to_time_format(segment.end)
            if text_only:
                transcription += f"{segment.text.strip()}\n"
            else:
                transcription += f"[{ts_start} --> {ts_end}] {segment.text}\n"
                sentence_idx += 1

    return transcription  # Remove trailing whitespace
