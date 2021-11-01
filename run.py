from omegaconf import DictConfig
import hydra


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == 'eval':
        import evaluate
        evaluate.evaluate(cfg.eval)
    elif cfg.mode == 'data_generation':
        import data_generation
        data_generation.data_generation(cfg.data_generation)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    main()
