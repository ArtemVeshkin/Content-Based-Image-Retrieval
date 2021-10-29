from omegaconf import DictConfig
import hydra


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == 'eval':
        import evaluate
        evaluate.evaluate(cfg.eval)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    main()
