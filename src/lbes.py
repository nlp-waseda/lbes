from transformers import HfArgumentParser

from lbes_config import LBESConfig
from lbes_trainer import LBESTrainer


def main() -> None:
    parser = HfArgumentParser(LBESConfig)
    lbes_config = parser.parse_args_into_dataclasses()[0]

    trainer = LBESTrainer(lbes_config)
    trainer.train()

    trainer.save_model(lbes_config.output_dir)


if __name__ == "__main__":
    main()
