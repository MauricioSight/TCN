import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score,
    log_loss, top_k_accuracy_score, balanced_accuracy_score
)

from metrics.base import InferenceMetrics
from utils.experiment_io import get_run_dir

class MultiClassificationMetrics(InferenceMetrics):
    def get_overall_metrics(self, y_true: pd.DataFrame, y_scores: np.ndarray) -> dict:
        """"
        Get overall metrics

        args:
            y_true: Data frame with labels
            y_scores: Model's output
            threshold: The threshold

        returns:
            dict of metrics
        """

        y_pred = np.argmax(y_scores, axis=1)
        y_prob = torch.tensor(np.array(y_scores)).softmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        self.logger.info('\n' + classification_report(y_true, y_pred))  # per-class P/R/F1 + macro/weighted

        top5 = top_k_accuracy_score(y_true, y_prob, k=5)
        ova_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        ce = log_loss(y_true, y_prob)

        # Save confusion matrix img
        run_dir = get_run_dir(self.config.get('run_id'))

        class_names = [str(i) for i in range(10)]
        _ = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=class_names
        )
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(run_dir / f"{self.config.get('phase')}_confusion_matrix.png", dpi=200)  # or .pdf / .svg
        plt.close()

        return {
            'acc': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'bal_acc': bal_acc,
            'top5': top5,
            'ova_auc': ova_auc,
            'ce': ce
        }


    def get_threshold(self, *args) -> float:
        """
        Define ou load threshold

        returns:
            threshold
        """
        pass