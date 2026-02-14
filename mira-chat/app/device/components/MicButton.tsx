"use client";

import { useState, useRef, useCallback, useEffect } from "react";

interface MicButtonProps {
  onResult: (text: string) => void;
  onTranscribing?: (active: boolean) => void;
  disabled?: boolean;
}

// Voice Activity Detection config
const SILENCE_THRESHOLD = 8; // RMS below this = silence (0-128 scale, lower = more lenient)
const SILENCE_DURATION_MS = 2200; // 2.2s of continuous silence to auto-stop
const MIN_SPEECH_DURATION_MS = 2000; // Must speak for at least 2s before silence detection kicks in
const MAX_RECORDING_MS = 30000; // 30s max recording

export default function MicButton({
  onResult,
  onTranscribing,
  disabled,
}: MicButtonProps) {
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const speechDetectedRef = useRef(false);
  const recordingStartRef = useRef<number>(0);
  const vadIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const maxTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stoppedRef = useRef(false);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (vadIntervalRef.current) clearInterval(vadIntervalRef.current);
      if (maxTimeoutRef.current) clearTimeout(maxTimeoutRef.current);
      audioContextRef.current?.close();
    };
  }, []);

  const stopAndTranscribe = useCallback(() => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;

    // Clean up VAD
    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }
    if (maxTimeoutRef.current) {
      clearTimeout(maxTimeoutRef.current);
      maxTimeoutRef.current = null;
    }

    setAudioLevel(0);

    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsListening(false);
  }, []);

  const startListening = useCallback(async () => {
    try {
      stoppedRef.current = false;
      speechDetectedRef.current = false;
      silenceStartRef.current = null;
      recordingStartRef.current = Date.now();

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Set up Web Audio API for volume monitoring
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.3;
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });

      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks and close audio context
        stream.getTracks().forEach((t) => t.stop());
        audioContext.close();

        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (blob.size < 100) return;

        setIsTranscribing(true);
        onTranscribing?.(true);

        try {
          const formData = new FormData();
          formData.append("audio", blob, "recording.webm");

          const res = await fetch("/api/voice/transcribe", {
            method: "POST",
            body: formData,
          });

          const data = await res.json();
          if (data.ok && data.text?.trim()) {
            onResult(data.text.trim());
          }
        } catch (err) {
          console.error("Whisper transcription failed:", err);
          fallbackWebSpeech();
        } finally {
          setIsTranscribing(false);
          onTranscribing?.(false);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsListening(true);

      // ── Voice Activity Detection loop ──
      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      vadIntervalRef.current = setInterval(() => {
        if (stoppedRef.current) return;

        analyser.getByteTimeDomainData(dataArray);

        // Calculate RMS volume
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const val = (dataArray[i] - 128) / 128;
          sum += val * val;
        }
        const rms = Math.sqrt(sum / dataArray.length) * 128;
        setAudioLevel(Math.min(rms / 40, 1)); // Normalize to 0-1 for visual

        const now = Date.now();
        const elapsed = now - recordingStartRef.current;

        if (rms > SILENCE_THRESHOLD) {
          // Speech detected
          speechDetectedRef.current = true;
          silenceStartRef.current = null;
        } else if (speechDetectedRef.current && elapsed > MIN_SPEECH_DURATION_MS) {
          // Silence after speech
          if (!silenceStartRef.current) {
            silenceStartRef.current = now;
          } else if (now - silenceStartRef.current > SILENCE_DURATION_MS) {
            // Silence long enough - auto-stop
            stopAndTranscribe();
          }
        }
      }, 80);

      // Max recording safety net
      maxTimeoutRef.current = setTimeout(() => {
        stopAndTranscribe();
      }, MAX_RECORDING_MS);
    } catch (err: unknown) {
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        window.alert(
          "Microphone access was denied.\n\nClick the lock icon in your address bar \u2192 set Microphone to Allow \u2192 then refresh."
        );
      } else {
        console.error("Microphone access failed:", err);
        fallbackWebSpeech();
      }
    }
  }, [onResult, onTranscribing, stopAndTranscribe]);

  const fallbackWebSpeech = useCallback(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0][0].transcript;
      if (transcript.trim()) onResult(transcript.trim());
      setIsListening(false);
    };
    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);

    recognition.start();
    setIsListening(true);
  }, [onResult]);

  const active = isListening || isTranscribing;

  return (
    <button
      type="button"
      onClick={isListening ? stopAndTranscribe : startListening}
      disabled={disabled || isTranscribing}
      className={`relative w-11 h-11 rounded-xl flex items-center justify-center transition-all shrink-0 disabled:opacity-30 ${
        isTranscribing
          ? "bg-amber-500 text-white shadow-md"
          : active
          ? "bg-red-500 text-white shadow-md mic-active-glow"
          : "bg-white border border-slate-200 text-slate-500 hover:text-teal-600 hover:border-teal-300 hover:bg-teal-50 shadow-sm"
      }`}
      title={
        isTranscribing
          ? "Transcribing..."
          : isListening
          ? "Listening... (will auto-stop)"
          : "Tap to speak"
      }
    >
      {/* Audio level ring (visible when listening) */}
      {isListening && (
        <div
          className="absolute inset-0 rounded-xl border-2 border-red-400 transition-all duration-100"
          style={{
            transform: `scale(${1 + audioLevel * 0.3})`,
            opacity: 0.3 + audioLevel * 0.5,
          }}
        />
      )}

      {isTranscribing ? (
        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
      ) : (
        <svg
          className="w-5 h-5 relative z-10"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      )}
    </button>
  );
}
