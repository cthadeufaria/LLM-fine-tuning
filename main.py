from model import Model


def main():
    # "Equall/SaulLM-54B-Instruct" (too large even to download / not fully tested) or 
    # "mradermacher/SaulLM-54B-Instruct-i1-GGUF" (too large to run / freezes on CPU / not fully tested) or
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = Model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

if __name__ == "__main__":
    main()