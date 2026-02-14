"use client";

import { useState, useRef, useCallback } from "react";

interface MicButtonProps {
  onResult: (text: string) => void;
  onTranscribing?: (active: boolean) => void;
  disabled?: boolean;
}

export default function MicButton({ onResult, onTranscribing, disabled }: MicButtonProps) {
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startListening = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
        stream.getTracks().forEach((t) => t.stop());

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
  }, [onResult, onTranscribing]);

  const stopListening = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsListening(false);
  }, []);

  const fallbackWebSpeech = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
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
      onClick={isListening ? stopListening : startListening}
      disabled={disabled || isTranscribing}
      className={`w-11 h-11 rounded-xl flex items-center justify-center transition-all shrink-0 disabled:opacity-30 ${
        isTranscribing
          ? "bg-amber-500 text-white shadow-md"
          : active
          ? "bg-red-500 text-white mic-pulse shadow-md"
          : "bg-white border border-slate-200 text-slate-500 hover:text-teal-600 hover:border-teal-300 hover:bg-teal-50 shadow-sm"
      }`}
      title={
        isTranscribing
          ? "Transcribing..."
          : isListening
          ? "Tap to stop"
          : "Tap to speak"
      }
    >
      {isTranscribing ? (
        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
      ) : (
        <svg
          className="w-5 h-5"
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
