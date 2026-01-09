from transformers import HfArgumentParser

from es_config import ESConfig
from es_trainer import ESTrainer


def main() -> None:
    parser = HfArgumentParser(ESConfig)
    es_config = parser.parse_args_into_dataclasses()[0]

    trainer = ESTrainer(es_config)
    trainer.train()

    trainer.save_model(es_config.output_dir)


if __name__ == "__main__":
    main()
