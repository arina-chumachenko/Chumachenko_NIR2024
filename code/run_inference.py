import os

def main():
    path = '/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/core/res_CR/'
    exps = [a for a in os.listdir('/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/core/res_CR/') if 'CR' in a]  # ['0044-res-duck_toy_CR', '0045-res-cat2_CR', '0048-res-dog_my_CR']
    exps = [(path + exp) for exp in exps]

    os.system("sh inference.sh 00006-res-dog6_CR 4000")
    os.system("sh inference.sh 00010-res-cat_CR 4000")
    os.system("sh inference.sh 00011-res-cat_CR 4000")
    os.system("sh inference.sh 00014-res-shiny_sneaker_CR 4000")
    os.system("sh inference.sh 00015-res-shiny_sneaker_CR 4000")


if __name__ == '__main__':
    main()
