import os

def main():
    # exp = 48
    # 42: dog6, validation_prompt="a S* in the jungle", dog6 '<dog6>' dog dog
    # 43: dog6, "a S* dressed as Spider-Man swings between tall buildings using webbing"
    # 44: duck_toy, "a S* adorned with dragon-scale armor, standing in the midst of a battle field with a determined gaze", duck_toy '<duck_toy>' toy duck_toy
    # 45: cat2, "a S* adorned with dragon-scale armor, standing in the midst of a battle field with a determined gaze", cat2 '<cat2>' cat cat
    # 46: cat2, "a S* dressed as Loki, with a tiny horned helmet and a green and gold outfit, standing on a throne in a grand Asgardian hall", cat2 '<cat2>' cat cat
    # 47: dog_my, "a S* dressed as a purple wizard on a desk in a medieval library"
    # 48: dog_my, "a S* in a halloween outfit", dog_my '<dog_my>' dog dog

    # dog6
    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 1 1 0 001 -C 'type_e'")
    # os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 001 -C 'type_e'")
    
    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 1 0 0 002 -C 'type_e'")
    # os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 002 -C 'type_e'")

    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 0 1 0 003 -C 'type_e'")
    # os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 006 -C 'type_e'")

    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 0 0 0 1 009 -C 'type_e'")
    os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 007 -C 'type_e'")

    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 0 0 1 005 -C 'type_e'")


    # # cat
    # os.system(f"sh train_cr.sh cat '<cat>' cat cat live 1 1 0 005 -C 'type_e'")
    # os.system(f"sh train_db.sh cat '<cat>' cat cat 005 -C 'type_e'")

    # os.system(f"sh train_cr.sh cat '<cat>' cat cat live 1 0 0 006 -C 'type_e'")
    # os.system(f"sh train_db.sh cat '<cat>' cat cat 006 -C 'type_e'")

    # os.system(f"sh train_cr.sh cat '<cat>' cat cat live 0 1 0 007 -C 'type_e'")
    os.system(f"sh train_db.sh cat '<cat>' cat cat 010 -C 'type_e'")

    # os.system(f"sh train_cr.sh cat '<cat>' cat cat live 0 0 0 008 -C 'type_e'")
    os.system(f"sh train_db.sh cat '<cat>' cat cat 011 -C 'type_e'")

    # os.system(f"sh train_cr.sh cat '<cat>' cat cat live 0 0 1 009 -C 'type_e'")
    # os.system(f"sh train_cr.sh cat '<cat>' cat cat live 0 0 0 1 013 -C 'type_e'")


    # shiny_sneaker
    # os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 1 1 0 009 -C 'type_e'")
    # os.system(f"sh train_db.sh shiny_sneaker '<sneaker>' sneaker sneaker 009 -C 'type_e'")
    
    # os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 1 0 0 010 -C 'type_e'")
    # os.system(f"sh train_db.sh shiny_sneaker '<sneaker>' sneaker sneaker 010 -C 'type_e'")

    # os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 0 1 0 011 -C 'type_e'")
    os.system(f"sh train_db.sh shiny_sneaker '<sneaker>' sneaker sneaker 014 -C 'type_e'")

    # os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 0 0 0 012 -C 'type_e'")
    os.system(f"sh train_db.sh shiny_sneaker '<sneaker>' sneaker sneaker 015 -C 'type_e'")

    # os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 0 0 1 013 -C 'type_e'")
    # os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 0 0 0 1 017 -C 'type_e'")



    # teapot
    # os.system(f"sh train_cr.sh teapot '<teapot>' teapot teapot object 0 0 1 013 -C 'type_e'")
    # os.system(f"sh train_db.sh teapot '<teapot>' teapot teapot 013 -C 'type_e'")

    # os.system(f"sh train_cr.sh teapot '<teapot>' teapot teapot object 0 0 0 014 -C 'type_e'")
    # os.system(f"sh train_db.sh teapot '<teapot>' teapot teapot 014 -C 'type_e'")

    # os.system(f"sh train_cr.sh teapot '<teapot>' teapot teapot object 1 1 0 015 -C 'type_e'")
    # os.system(f"sh train_db.sh teapot '<teapot>' teapot teapot object 015 -C 'type_e'")

    # os.system(f"sh train_cr.sh teapot '<teapot>' teapot teapot object 1 0 0 016 -C 'type_e'")
    # os.system(f"sh train_db.sh teapot '<teapot>' teapot teapot object 016 -C 'type_e'")
    
    # os.system(f"sh train_cr.sh teapot '<teapot>' teapot teapot object 0 1 0 017 -C 'type_e'")


    # os.system(f"sh train_cr.sh cat2 '<cat2>' cat cat live 007 -C 'type_e'")
    # os.system(f"sh train_db.sh cat2 '<cat2>' cat cat live 007 -C 'type_e'")

    # os.system(f"sh train_cr.sh duck_toy '<duck_toy>' toy duck_toy object 005 -C 'type_e'")
    # os.system(f"sh train_db.sh duck_toy '<duck_toy>' toy duck_toy object 005 -C 'type_e'")

    # os.system(f"sh train_cr.sh vase '<vase>' vase vase object 005 -C 'type_e'")
    # os.system(f"sh train_db.sh vase '<vase>' vase vase object 005 -C 'type_e'")
    
    # os.system(f"sh train_cr.sh backpack '<backpack>' backpack backpack 05 -C 'type_e'")
    # os.system(f"sh train_db.sh backpack '<backpack>' backpack backpack 05 -C 'type_e'")
    
    # os.system(f"sh train_cr.sh berry_bowl '<bowl>' bowl bowl object 007 -C 'type_e'")
    # os.system(f"sh train_db.sh berry_bowl '<bowl>' bowl bowl 007 -C 'type_e'")
    
    # os.system(f"sh train_cr.sh backpack_dog '<backpack>' backpack backpack object 010 -C 'type_e'")
    # os.system(f"sh train_db.sh backpack_dog '<backpack>' backpack backpack 010 -C 'type_e'")


if __name__ == '__main__':
    main()