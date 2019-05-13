class Hero():

    def __init__(self,name,age):
        self.name = name
        self.age = age
        my = age + 10
    
    def who(self):
        print("Name : ",self.name)
        print("Age : ",self.age)

if __name__ == "__main__":
    hero = Hero("kabir",4)
    hero.who()
    print(hero.name)
