import scipy.io

class STINK_Loader:
    def __init__(self,path_to_dataset):
        self.dataset = scipy.io.loadmat(path_to_dataset)

    def get_dataset(self):
        return self.dataset


if __name__ == "__main__":
    path_to_file = "../../datasets/STINK/all_locations_v2.mat"
    s_loader = STINK_Loader(path_to_file)
    stink_dataset = s_loader.get_dataset()
    print("finished")
