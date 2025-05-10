from torchvision.datasets import MNIST

TARGET_SIZE = 224
Y_FILE = "./data/mod/y.csv"

def main():
    data = MNIST(root="./data/raw", download=True)

    with open(Y_FILE, "w") as file:
        for i, d in enumerate(data):
            img, y = data.__getitem__(i)
            img = img.resize((TARGET_SIZE, TARGET_SIZE))
            img.save(f"./data/mod/{i}.jpg")
            file.write(f"{y}\n")

if __name__ == "__main__":
    main()
