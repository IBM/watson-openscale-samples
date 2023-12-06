# Evaluating prompt template asset of different task types
This document explains how to configure evaluations of prompt template with different task types.


The available task ids that we have are are
1. summarization - This id is applicable for tasks related to summarization.
2. classification - Intended for tasks related to classification.
3. question_answering - Recommended for tasks involving question answering
4. generation - This identifier is suitable for tasks involving content generation.
5. extraction - This identifier is designated for tasks related to entity extraction


Users can set up a genarative ai quality monitor across all task types except for classification. In the case of classification, users have the option to configure the quality monitor. Drift V2 monitor can be configured with production subscription.


# Sample congiguration for Generative ai quality monitor is:

    "generative_ai_quality": {
        "parameters": {

            "min_sample_size": 10,
            "metrics_configuration":{
                
            "bleu": {
                "max_order": 4,
                "smooth": "false"
            },
            "cosine_similarity": {},
            "hap_score": {
                "record_level_max_score": 0.5
            },
            "jaccard_similarity": {},
            "meteor": {
                "alpha": 0.9,
                "beta": 3,
                "gamma": 0.5
            },
            "normalized_f1": {},
            "normalized_precision": {},
            "normalized_recall": {},
            "rouge_score": {
                "use_aggregator": "true",
                "use_stemmer": "true"
            },
            "sari": {},
            "pii": {
              "record_level_max_score": 0.5
             }
                    
            }
        }
    }


# Sample congiguration for quality monitor is:

    "quality": {
        "thresholds": [
            {
                "metric_id": "accuracy",
                "type": "lower_limit",
                "value": 0.8
            },
            {
                "metric_id": "weighted_false_positive_rate",
                "type": "upper_limit",
                "value": 0.5
            },
            {
                "metric_id": "weighted_f_measure",
                "type": "lower_limit",
                "value": 0.8
            },
            {
                "metric_id":"label_skew",
                "type":"upper_limit",
                "value":0.5
            },
            {
                "metric_id":"label_skew",
                "type":"lower_limit",
                "value":-0.5
            },
            {
                "metric_id": "matthews_correlation_coefficient",
                "type": "lower_limit",
                "value": 0.8
            },
            {
                "metric_id": "weighted_true_positive_rate",
                "type": "lower_limit",
                "value": 0.8
            },
            {
                "metric_id": "weighted_precision",
                "type": "lower_limit",
                "value": 0.7
            },
            {
                "metric_id": "weighted_recall",
                "type": "lower_limit",
                "value": 0.7
            }
        ],
        "parameters": {
            "evaluation_definition": {
                "threshold": 0.8
            },
            "min_feedback_data_size": 10
        }
    }


# Sample congiguration for Drift V2 monitor is:

    "drift_v2": {
        "thresholds": [
            {
                "metric_id": "confidence_drift_score",
                "type": "upper_limit",
                "value": 0.05
            },
            {
                "metric_id": "prediction_drift_score",
                "type": "upper_limit",
                "value": 0.05
            },
            {
                "metric_id": "input_metadata_drift_score",
                "specific_values": [
                    {
                        "applies_to": [
                            {
                                "type": "tag",
                                "value": "subscription",
                                "key": "field_type"
                            }
                        ],
                        "value": 0.05
                    }
                ],
                "type": "upper_limit"
            },
            {
                "metric_id": "output_metadata_drift_score",
                "specific_values": [
                    {
                        "applies_to": [
                            {
                                "type": "tag",
                                "value": "subscription",
                                "key": "field_type"
                            }
                        ],
                        "value": 0.05
                    }
                ],
                "type": "upper_limit"
            }
        ],
        "parameters": {
            "min_samples": 10,
            "train_archive": True
        }
    }

