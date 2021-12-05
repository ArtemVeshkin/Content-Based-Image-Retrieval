from omegaconf import DictConfig
import hydra


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == 'eval':
        from CBIR.evaluate import evaluate
        evaluate(cfg.eval)
    elif cfg.mode == 'data_generation':
        from CBIR.data_generation import data_generation
        data_generation(cfg.data_generation)
    elif cfg.mode == 'fit_VAE':
        from CBIR.fit_VAE import fit_VAE
        fit_VAE(cfg.fit_VAE)
    elif cfg.mode == 'extractor_visualization':
        from CBIR.extractor_visualization import extractor_visualization
        extractor_visualization(cfg.extractor_visualization)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    main()
