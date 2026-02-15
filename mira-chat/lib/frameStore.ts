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
  private totalFrames: number = 0;
  private startTime: number = Date.now();

  setFrame(frame: FrameData) {
    this.latestFrame = frame;
    this.totalFrames++;
  }

  getFrame(): FrameData | null {
    return this.latestFrame;
  }

  hasFrame(): boolean {
    return this.latestFrame !== null;
  }

  getTotalFrames(): number {
    return this.totalFrames;
  }

  getUptimeSeconds(): number {
    return Math.floor((Date.now() - this.startTime) / 1000);
  }

  reset() {
    this.latestFrame = null;
    this.totalFrames = 0;
    this.startTime = Date.now();
  }
}

// Singleton instance
const frameStore = new FrameStore();

export default frameStore;
export type { FrameData };
