#Arquivo utilizado para baixar o dataset SVHN do torch vision

from torchvision import datasets, transforms #Necessário ter o torchvision instalado

marcador = []

def baixar(tipo,t):
    return datasets.SVHN(
        root = "./DataSet",
        split = tipo,
        transform = t,
        download=True
    )

def main():
    transformar = transforms.ToTensor()

    dataset_treino = baixar("train",transformar)
    dataset_teste = baixar("test",transformar)

    marcador.append("download concluído!")
    
main()