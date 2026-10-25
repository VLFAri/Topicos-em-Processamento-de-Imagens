from torchvision import datasets, transforms


def baixar(tipo, transformacao):
    print(f"Iniciando download do conjunto '{tipo}'...")

    dataset = datasets.SVHN(
        root="./DataSet",
        split=tipo,
        transform=transformacao,
        download=True
    )

    print(f"Conjunto '{tipo}' carregado com sucesso!")
    return dataset


def main():
    transformar = transforms.ToTensor()

    dataset_treino = baixar("train", transformar)
    dataset_teste = baixar("test", transformar)

    print("\nDownload concluído!")
    print(f"Imagens de treinamento: {len(dataset_treino)}")
    print(f"Imagens de teste: {len(dataset_teste)}")


if __name__ == "__main__":
    main()