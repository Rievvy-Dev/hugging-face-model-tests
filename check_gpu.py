# -*- coding: utf-8 -*-
"""
Verifica se o PyTorch enxerga a GPU (CUDA).
Se nao enxergar, mostra como instalar PyTorch com suporte a CUDA.
"""
import sys

def main():
    print("Python em uso:", sys.executable)
    print()
    try:
        import torch
    except ImportError:
        print("PyTorch nao esta instalado.")
        print("Instale com o MESMO Python acima:")
        print("  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return 1

    cuda_ok = torch.cuda.is_available()
    print("PyTorch:", torch.__version__)
    print("CUDA disponivel?", cuda_ok)
    if cuda_ok:
        print("Device:", torch.cuda.get_device_name(0))
        print("Versao CUDA (driver):", torch.version.cuda or "N/A")
        return 0

    print("\n>>> GPU nao detectada (voce tem PyTorch CPU-only).")
    print("Para usar GPU, use o MESMO Python que rodou este script:")
    print("")
    print("  1. Desinstalar o torch atual:")
    print("     python -m pip uninstall torch torchvision torchaudio -y")
    print("")
    print("  2. Instalar PyTorch com CUDA 12.4:")
    print("     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("")
    print("  (Se der erro, tente cu121 ou cu118 no lugar de cu124.)")
    print("")
    print("  3. Rodar de novo: python check_gpu.py")
    return 1

if __name__ == "__main__":
    sys.exit(main())
