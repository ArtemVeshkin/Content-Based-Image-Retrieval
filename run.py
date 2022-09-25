from omegaconf import DictConfig
import hydra


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == 'CBIR_test':
        from CBIR.scenarios.CBIR_test import CBIR_test
        CBIR_test(cfg.CBIR_test)
    elif cfg.mode == 'eval':
        from CBIR.scenarios.evaluate import evaluate
        evaluate(cfg.eval)
    elif cfg.mode == 'data_generation':
        from CBIR.scenarios.data_generation import data_generation
        data_generation(cfg.data_generation)
    elif cfg.mode == 'fit_VAE':
        from CBIR.scenarios.fit_VAE import fit_VAE
        fit_VAE(cfg.fit_VAE)
    elif cfg.mode == 'fit_scalenet':
        from CBIR.scenarios.fit_scalenet import fit_scalenet
        fit_scalenet(cfg.fit_scalenet)
    elif cfg.mode == 'fit_contrastive_extractor':
        from CBIR.scenarios.fit_contrastive_extractor import fit_contrastive_extractor
        fit_contrastive_extractor(cfg.fit_contrastive_extractor)
    elif cfg.mode == 'extractor_visualization':
        from CBIR.scenarios.extractor_visualization import extractor_visualization
        extractor_visualization(cfg.extractor_visualization)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    main()
