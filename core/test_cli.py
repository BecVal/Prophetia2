import cli_predictor

class MockQuestionary:
    def __init__(self, ret_val):
        self.ret_val = ret_val
    def ask(self):
        return self.ret_val

def mock_select(*args, **kwargs):
    return MockQuestionary("E1")

def mock_autocomplete(msg, *args, **kwargs):
    if "Local" in msg:
        return MockQuestionary("Arsenal")
    return MockQuestionary("Chelsea")

def mock_text(msg, *args, **kwargs):
    if "Local [1]" in msg:
        return MockQuestionary("2.10")
    elif "Empate [X]" in msg:
        return MockQuestionary("3.40")
    elif "Visitante [2]" in msg:
        return MockQuestionary("3.50")
    elif "Bankroll" in msg:
        return MockQuestionary("1000")
    elif "Kelly" in msg:
        return MockQuestionary("0.20")
    elif "Lesiones" in msg:
        return MockQuestionary("0")
    return MockQuestionary("0")

cli_predictor.questionary.select = mock_select
cli_predictor.questionary.autocomplete = mock_autocomplete
cli_predictor.questionary.text = mock_text

if __name__ == '__main__':
    cli_predictor.main()
