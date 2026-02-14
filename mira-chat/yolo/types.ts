export interface Detection {
  label: string;
  confidence: number;
  /** Bounding box in pixel coords (relative to frame) */
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Color map for HUD-style bounding boxes */
export const LABEL_COLORS: Record<string, string> = {
  person: "rgba(120,255,200,0.9)",
  cup: "rgba(100,200,255,0.9)",
  bottle: "rgba(100,200,255,0.9)",
  bowl: "rgba(100,200,255,0.9)",
  wine_glass: "rgba(100,200,255,0.9)",
  chair: "rgba(180,160,255,0.9)",
  couch: "rgba(180,160,255,0.9)",
  bed: "rgba(180,160,255,0.9)",
  dining_table: "rgba(180,160,255,0.9)",
  cell_phone: "rgba(255,200,100,0.9)",
  remote: "rgba(255,200,100,0.9)",
  laptop: "rgba(255,200,100,0.9)",
  tv: "rgba(255,200,100,0.9)",
  book: "rgba(255,180,180,0.9)",
  clock: "rgba(255,180,180,0.9)",
  scissors: "rgba(255,180,180,0.9)",
  potted_plant: "rgba(160,255,160,0.9)",
};

export const DEFAULT_COLOR = "rgba(120,255,200,0.7)";

export function getColor(label: string): string {
  return LABEL_COLORS[label.replace(" ", "_")] || DEFAULT_COLOR;
}

// COCO 80-class labels (YOLOv8 output order)
export const COCO_LABELS = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
  "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];
