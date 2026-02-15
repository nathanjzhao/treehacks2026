// Shared frame storage for streaming between API routes
// This module provides a singleton in-memory store for the latest video frame

interface FrameData {
  data: Buffer;
  timestamp: string;
  width: number;
  height: number;
  frameNumber: number;
}

class FrameStore {
  private latestFrame: FrameData | null = null;

  setFrame(frame: FrameData) {
    this.latestFrame = frame;
  }

  getFrame(): FrameData | null {
    return this.latestFrame;
  }

  hasFrame(): boolean {
    return this.latestFrame !== null;
  }
}

// Singleton instance
const frameStore = new FrameStore();

export default frameStore;
