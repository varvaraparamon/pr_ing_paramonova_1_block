import argparse

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def main():
    parser = argparse.ArgumentParser(description="Классификация изображения")
    parser.add_argument("image", help="путь к файлу изображения")
    parser.add_argument(
        "--topk", type=int, default=5, help="сколько лучших классов показать"
    )
    args = parser.parse_args()

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    batch = weights.transforms()(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[0]

    top = probs.topk(args.topk)
    for prob, idx in zip(top.values, top.indices):
        print(f"{prob.item():.4f}  {weights.meta['categories'][idx]}")


if __name__ == "__main__":
    main()
