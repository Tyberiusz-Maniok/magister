from torchvision.datasets import MNIST

TARGET_SIZE = 228

def main():
    data = MNIST(root="./data/raw", download=True)

    # for i, d in enumerate(data):
    #     img = d.resize((TARGET_SIZE, TARGET_SIZE))
    #     img.save(f"./data/mod/{i}.jpg")

    img = data.__getitem__(0)[0]
    img = img.resize((TARGET_SIZE, TARGET_SIZE))
    img.save("./data/mod/1.jpg")



if __name__ == "__main__":
    main()
