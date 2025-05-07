from torchvision.datasets import MNIST

TARGET_SIZE = 228

def main():
    data = MNIST(root="./data/raw", download=True)

    for i, d in enumerate(data):
        img = data.__getitem__(i)[0]
        img = img.resize((TARGET_SIZE, TARGET_SIZE))
        img.save(f"./data/mod/{i}.jpg")

if __name__ == "__main__":
    main()
