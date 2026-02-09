import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch

from metrics.base import InferenceMetrics
from metrics.timming_metrics import get_resource_metrics


class AEROMetrics(InferenceMetrics):
    def get_overall_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray, threshold: float = None) -> dict:
        """"
        Get overall metrics

        args:
            y_true: Data frame with labels
            y_scores: Model's output

        returns:
            dict of metrics
        """

        _, y_scores = y_pred

        y_true['scores'] = y_scores
        y_true_benign = y_true[y_true["label"] == 'Normal']
        mean = y_true_benign["scores"].mean()
        std = y_true_benign["scores"].std()
        
        y_true_labels = y_true['label'].values
        y_true_binary = [0 if l == 'Normal' else 1 for l in y_true_labels]

        aucroc = roc_auc_score(y_true_binary, y_scores)
        aucroc_per_attack = self.__roc_auc_score_each_attack(y_true_labels, y_scores)

        if not threshold:
            threshold = self.__get_aero_threshold()

        result = self.__get_overall_metrics(y_true_binary, y_scores > threshold)

        tpr_per_attack = self.__get_tpr_per_attack(y_true_labels, y_scores > threshold)

        overall_metrics = {'AUCROC': aucroc, **result, 'optimal_threshold': threshold, 'mean': mean, 'std': std}
        metrics_serializable = {k: float(v) for k, v in overall_metrics.items()}
        metrics_serializable['tpr_per_attack'] = tpr_per_attack
        metrics_serializable['aucroc_per_attack'] = aucroc_per_attack

        if 'model' in self.context:
            model           = self.context['model']
            train_data      = self.context['train_data']
            resource_metrics = get_resource_metrics(model, torch.from_numpy(train_data[0][:1]))
            metrics_serializable['resource_metrics'] = resource_metrics

        self.logger.info(f"Metrics \n{json.dumps(metrics_serializable, indent=4)}")

        return metrics_serializable


    def __get_tpr_per_attack(self, y_labels, y_scores):
        aux_df = pd.DataFrame({'Label':y_labels,'prediction':y_scores})
        total_per_label = aux_df['Label'].value_counts().to_dict()
        correct_predictions_per_label = aux_df.query('Label != "Normal" and prediction == True').groupby('Label').size().to_dict()
        tpr_per_attack = {}
        for attack_label, total in total_per_label.items():
            if attack_label == 'Normal':
                continue
            tp = correct_predictions_per_label[attack_label] if attack_label in correct_predictions_per_label else 0
            tpr = tp/total
            tpr_per_attack[attack_label] = tpr
        
        tpr_per_attack['Normal'] = (aux_df[(aux_df['Label'] == 'Normal') & (aux_df['prediction'] == False)].shape[0] / 
                                    aux_df[(aux_df['Label'] == 'Normal')].shape[0])
        
        tpr_per_attack['Attack'] = (aux_df[(aux_df['Label'] != 'Normal') & (aux_df['prediction'] != False)].shape[0] / 
                                    aux_df[(aux_df['Label'] != 'Normal')].shape[0])

        return tpr_per_attack


    def __get_overall_metrics(self, y_true, y_scores):
        tn, fp, fn, tp = confusion_matrix(y_true, y_scores).ravel()
        acc = (tp+tn)/(tp+tn+fp+fn)
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        precision = tp/(tp+fp)
        f1 = (2*tpr*precision)/(tpr+precision)
        return {'Accuracy':acc,'TPR':tpr,'FPR':fpr,'Precision':precision,'F1-score':f1}


    def __roc_auc_score_each_attack(self, y_true, y_scores):
        """
        Compute AUC-ROC for each attack type in the dataset.
        Assumes y_true contains labels and y_scores contains predictions.
        """
        unique_labels = set(y_true)
        auc_scores = {}
        
        for label in unique_labels:
            if label == 'Normal':
                continue  # Skip normal class
            attack_y_true = [1 if l == label else 0 for l in y_true if l == label or l == 'Normal']
            attack_y_scores = [y_scores[i] for i in range(len(y_true)) if y_true[i] == label or y_true[i] == 'Normal']
            auc_score = roc_auc_score(attack_y_true, attack_y_scores)
            auc_scores[label] = float(auc_score)
        
        return auc_scores


    def get_threshold(self, *args) -> float:
        """
        Define ou load threshold

        returns:
            threshold
        """
        pass

    
    def __get_aero_threshold(self):
        """
        Determine anomaly detection threshold τ using validation set.
        
        Args:
            val_loader: DataLoader for validation data.
            percentile: desired percentile for threshold τ.
        
        Returns:
            threshold τ (float)
        """
        percentile = self.config.get('metrics', {}).get('percentile', 95)

        model_inference = self.context['model_inference']
        model           = self.context['model']
        train_data      = self.context['train_data']
        
        y = train_data[1].reset_index(drop=True)
        benign_idx = y[y['label'] == 'Normal'].index
        _, (_, y_scores), _ = model_inference.inference(model, train_data[0][benign_idx], y.iloc[benign_idx])
        y_scores = np.array(y_scores)

        # Step 4: Determine threshold τ = P(l, p)
        threshold = np.percentile(y_scores, percentile)

        return threshold