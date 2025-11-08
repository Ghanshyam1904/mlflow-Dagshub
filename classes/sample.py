class Chatbook:
    def __init__(self):
        self.username = ""
        self.password = ""
        self.loggdin = False
        self.menu()

    def menu(self):
        user_input = input("""Welcome to chatbook
            Press 1. for sign up
            Press 2. for sign in
            Press 3. to write post
            Press 4. to msg friend
            Press any key to exit""")

        if user_input == "1":
            pass
        elif user_input == "2":
            pass
        elif user_input == "3":
            pass
        elif user_input == "4":
            pass
        else:
            exit()

obj = Chatbook()