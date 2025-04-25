from torchvision.datasets import MNIST

def main():
    data = MNIST(root="./data/raw", download=True)

if __name__ == "__main__":
    main()
