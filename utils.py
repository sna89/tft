import pickle


def save_to_pickle(file, path):
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pickle(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file
