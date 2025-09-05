from model import ModelWrapper
from trainer import UnsupervisedTrainer


def main():
    # "Equall/SaulLM-54B-Instruct" (too large even to download / not fully tested)
    # "mradermacher/SaulLM-54B-Instruct-i1-GGUF" (too large to run / freezes on CPU / not fully tested)
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    wrapper = ModelWrapper("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    trainer = UnsupervisedTrainer(wrapper)
    trainer.train_model()


if __name__ == "__main__":
    main()