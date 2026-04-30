export interface EpochLog {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  lr: number;
  timestamp: string;
}

export interface TrainingMetrics {
  epochs: EpochLog[];
  status: string;
}

export interface PredictionResult {
  result: 'APROVADA' | 'DEFEITO';
  label: number;
  confidence: number;
  probability_defect: number;
  probability_ok: number;
  filename: string;
}

export interface ConfusionMatrix {
  matrix: number[][];
  labels: string[];
}

export interface EvaluationResults {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: ConfusionMatrix;
  business_impact: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
    note: string;
  };
  total_test_samples: number;
}
