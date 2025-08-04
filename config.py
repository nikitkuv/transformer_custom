from dataclasses import dataclass


@dataclass
class Config:

    # data paths
    train_data_path: str = 'data/data_multi_30k_train.csv'
    test_data_path: str = 'data/data_multi_30k_test.csv'
    batch_size: int = 32

    # tokenizers setup
    source_language: str = 'de'
    target_language: str = 'en'
    max_length: int = 32
    source_tokenizer: str = 'bert-base-uncased'
    target_tokenizer: str = 'dbmdz/bert-base-german-uncased'

    # model setup
    hidden_size: int = 128    
    number_of_attention_heads: int = 8     
    feed_forward_intermediate_size: int = 512    
    number_of_blocks: int = 6
    dropout_rate: float = 0.1
    number_of_epochs: int = 18

    # optimizer setup
    ema_decay: float = 0.999
    learning_rate = 5e-4
    warmup_steps:int = 5000
    max_training_steps: int = 22000
    min_learming_rate: float = 1e-6

    use_wandb: bool = False
    wandb_project_name: str = 'train_trainsformer'
    wanndb_run_name: str = 'run1'
    