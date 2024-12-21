import os

def main():

    # dog6
    os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 1 1 0 0 001 -C 'type_e'")
    os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 001 -C 'type_e'")
    
    os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 1 0 0 0 002 -C 'type_e'")
    os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 002 -C 'type_e'")

    os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 0 1 0 0 003 -C 'type_e'")
    os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 006 -C 'type_e'")

    os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 0 0 0 1 009 -C 'type_e'")
    os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 007 -C 'type_e'")

    os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 0 0 1 005 -C 'type_e'")


    # # cat
    os.system(f"sh train_cr.sh cat '<cat>' cat cat live 1 1 0 005 -C 'type_e'")
    os.system(f"sh train_db.sh cat '<cat>' cat cat 005 -C 'type_e'")

    os.system(f"sh train_cr.sh cat '<cat>' cat cat live 1 0 0 006 -C 'type_e'")
    os.system(f"sh train_db.sh cat '<cat>' cat cat 006 -C 'type_e'")

    os.system(f"sh train_cr.sh cat '<cat>' cat cat live 0 1 0 007 -C 'type_e'")
    os.system(f"sh train_db.sh cat '<cat>' cat cat 010 -C 'type_e'")

    os.system(f"sh train_cr.sh cat '<cat>' cat cat live 0 0 0 008 -C 'type_e'")
    os.system(f"sh train_db.sh cat '<cat>' cat cat 011 -C 'type_e'")

    os.system(f"sh train_cr.sh cat '<cat>' cat cat live 0 0 1 009 -C 'type_e'")
    os.system(f"sh train_cr.sh cat '<cat>' cat cat live 0 0 0 1 013 -C 'type_e'")


    # shiny_sneaker
    os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 1 1 0 009 -C 'type_e'")
    os.system(f"sh train_db.sh shiny_sneaker '<sneaker>' sneaker sneaker 009 -C 'type_e'")
    
    os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 1 0 0 010 -C 'type_e'")
    os.system(f"sh train_db.sh shiny_sneaker '<sneaker>' sneaker sneaker 010 -C 'type_e'")

    os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 0 1 0 011 -C 'type_e'")
    os.system(f"sh train_db.sh shiny_sneaker '<sneaker>' sneaker sneaker 014 -C 'type_e'")

    os.system(f"sh train_cr.sh shiny_sneaker '<sneaker>' sneaker sneaker object 0 0 0 012 -C 'type_e'")
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


if __name__ == '__main__':
    main()
