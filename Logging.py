import os


class Logging:

    file_name: str = ""

    def __init__(self, file_name: str):
        self.file_name = file_name
        if (not os.path.exists("Logs")):
            os.mkdir("Logs")
            print("Directory ", "'Logs'",  " Created ")
        else:
            print("Directory ", "'Logs'",  " already exists")

    def log(self, text: str):
        with open("./Logs/"+self.file_name, "w") as file:
            file.write(text)
