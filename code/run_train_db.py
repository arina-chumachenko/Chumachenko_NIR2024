import os

def main():
    os.system("sh train_db.sh dog6 '<dog6>' dog dog 40 -C 'type_e'")
    # os.system("sbatch train.sh berry_bowl bowl sks")
    # os.system("sbatch train.sh backpack_dog backpack sks")
    # os.system("sbatch train.sh can can sks")
    # os.system("sbatch train.sh cat cat sks")
    # os.system("sbatch train.sh clock clock sks")
    # os.system("sbatch train.sh monster_toy toy sks")

if __name__ == '__main__':
    main()
