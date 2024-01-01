# Evaluating prompt template asset of different task types
This document explains how to configure evaluations of prompt template with different task types.


The available task ids that we have are are
1. summarization - This id is applicable for tasks related to summarization.
2. classification - Intended for tasks related to classification.
3. question_answering - Recommended for tasks involving question answering
4. generation - This identifier is suitable for tasks involving content generation.
5. extraction - This identifier is designated for tasks related to entity extraction


Users can set up a generative ai quality monitor across all task types except for classification. In the case of classification, users have the option to configure the quality monitor. Drift V2 monitor can be configured with production subscription.



# Sample configuration for Generative ai quality monitor:

All metrics for generative ai quality are not supported for all task types. Hence configurations matching for different task types are listed down separately.


## Sample configuration for `summarization` task type

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

## Sample configuration for `generation` task type

  "generative_ai_quality": {
      "thresholds": [
        {
          "metric_id": "rouge1",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "rouge2",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "rougel",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "rougelsum",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "normalized_f1",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "normalized_precision",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "normalized_recall",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "pii",
          "type": "upper_limit",
          "value": 0
        },
        {
          "metric_id": "hap_score",
          "type": "upper_limit",
          "value": 0
        },
        {
          "metric_id": "pii_input",
          "type": "upper_limit",
          "value": 0
        },
        {
          "metric_id": "hap_input_score",
          "type": "upper_limit",
          "value": 0
        },
        {
          "metric_id": "meteor",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "bleu",
          "type": "lower_limit",
          "value": 0.8
        },
        {
          "metric_id": "flesch_reading_ease",
          "type": "lower_limit",
          "value": 60
        }
      ],
      "parameters": {
        "metrics_configuration": {
          "pii": {
            "record_level_max_score": 0.5
          },
          "hap_score": {
            "record_level_max_score": 0.5
          },
          "pii_input": {
            "record_level_max_score": 0.5
          },
          "hap_input_score": {
            "record_level_max_score": 0.5
          },
          "bleu": {
            "max_order": 4,
            "smooth": false
          },
          "flesch": {},
          "meteor": {
            "alpha": 0.9,
            "beta": 3,
            "gamma": 0.5
          },
          "normalized_recall": {},
          "normalized_f1": {},
          "rouge_score": {
            "use_aggregator": true,
            "use_stemmer": false
          },
          "normalized_precision": {}
        }
      }
    }

## Sample configuration for `question_answering` task type

  "generative_ai_quality": {
    "thresholds": [
      {
        "metric_id": "rouge1",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "rouge2",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "rougel",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "rougelsum",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "exact_match",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "pii",
        "type": "upper_limit",
        "value": 0
      },
      {
        "metric_id": "hap_score",
        "type": "upper_limit",
        "value": 0
      },
      {
        "metric_id": "pii_input",
        "type": "upper_limit",
        "value": 0
      },
      {
        "metric_id": "hap_input_score",
        "type": "upper_limit",
        "value": 0
      },
      {
        "metric_id": "bleu",
        "type": "lower_limit",
        "value": 0.8
      }
    ],
    "parameters": {
      "metrics_configuration": {
        "pii": {
          "record_level_max_score": 0.5
        },
        "hap_score": {
          "record_level_max_score": 0.5
        },
        "pii_input": {
          "record_level_max_score": 0.5
        },
        "hap_input_score": {
          "record_level_max_score": 0.5
        },
        "exact_match": {},
        "bleu": {
          "max_order": 4,
          "smooth": false
        },
        "rouge_score": {
          "use_aggregator": true,
          "use_stemmer": true
        }
      }
    }
  }

## Sample configuration for `extraction` task type

  "generative_ai_quality": {
    "thresholds": [
      {
        "metric_id": "rouge1",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "rouge2",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "rougel",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "rougelsum",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "micro_f1",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "macro_f1",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "micro_precision",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "macro_precision",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "micro_recall",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "macro_recall",
        "type": "lower_limit",
        "value": 0.8
      },
      {
        "metric_id": "exact_match",
        "type": "lower_limit",
        "value": 0.8
      }
    ],
    "parameters": {
      "metrics_configuration": {
        "exact_match": {},
        "multi_label_metrics": {},
        "rouge_score": {
          "use_aggregator": true,
          "use_stemmer": true
        }
      }
    }
  }

# Sample configuration for quality monitor is:

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


# Sample configuration for Drift V2 monitor is:

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

