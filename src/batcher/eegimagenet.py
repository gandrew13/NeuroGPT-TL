from .base import *


# TODO: refactor, reuse code from .base
class EEGImageNetDataset(Dataset):
    def __init__(self, loaded_data, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, population_mean=0, population_std=1, gpt_only=False, normalization=True, start_samp_pnt=-1):
        
        granularity = "coarse"
        
        #self.labels = loaded["labels"]
        #self.images = loaded["images"]

        chosen_data = loaded_data
        
        if granularity == 'coarse':
            self.data = [i for i in chosen_data if i['granularity'] == 'coarse']
        elif granularity == 'all':
            self.data = chosen_data
        else:
            fine_num = int(granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [i for i in chosen_data if i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]
            
        print(self.data[10]["eeg_data"].size(), self.data[10]["granularity"], self.data[10]["subject"], self.data[10]["label"], self.data[10]["image"])
        #print("Number of samples loaded: ", len(self.data))
        [print(entry['granularity']) for entry in self.data if entry['granularity'] != "coarse"]

        # one-hot encoding
        self.labels = []
        [self.labels.append(entry['label']) for entry in self.data if entry['label'] not in self.labels and entry['granularity'] == 'coarse']

        #print(torch.tensor(self.labels))
        #print(len(set(self.labels)))

        #self.one_hots = f.one_hot(torch.tensor(self.labels), num_classes = len(self.labels))
        
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only=gpt_only
        self.start_samp_pnt = start_samp_pnt

    def __len__(self):
        return len(self.data)
        # return 32 * 100 # for testing purposes

    def __getitem__(self, idx):
        data = self.data[idx]['eeg_data']
        data = data[:, 40:440]
        data = data.numpy()

        # keep only 22 channels, as in the original paper (or use a mapping NN)
        #data = self.map_channels(data)

        # reorder channels
        #data = self.reorder_channels(data)

        # used specifically to remove channels not found in the Alljoined1 dataset
        # data = Alljoined1.remove_channels()

        # data = self.__get_dummy_inputs() # for testing purposes

        label_idx = self.labels.index(self.data[idx]['label'])
        if label_idx == -1:
            print("Error! Label not found!")
            return

        return EEGDataset.preprocess_sample(data, seq_len=self.num_chunks, labels=label_idx)  
    
    def __get_dummy_inputs(self) -> torch.tensor:
        return torch.rand([22, 500]) # nr_channels, time_series_length


    def reorder_channels(self, data):
        chann_labels = {'FP1': 0, 'FP2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9, 'F7': 10, 'F8': 11, 'T3': 12, 'T4': 13, 'T5': 14, 'T6': 15, 'FZ': 16, 'CZ': 17, 'PZ': 18, 'OZ': 19, 'T1': 20, 'T2': 21}
        reorder_labels = {'FP1': 0, 'FP2': 1, 'F7': 2, 'F3': 3, 'FZ': 4, 'F4': 5, 'F8': 6, 'T1': 7, 'T3': 8, 'C3': 9, 'CZ': 10, 'C4': 11, 'T4': 12, 'T2': 13, 'T5': 14, 'P3': 15, 'PZ': 16, 'P4': 17, 'T6': 18, 'O1': 19, 'OZ': 20, 'O2': 21}

        reordered = np.zeros_like(data)
        for label, target_idx in reorder_labels.items():
            mapped_idx = chann_labels[label]
            reordered[target_idx, :] = data[mapped_idx, :]
        
        return reordered

    def split_chunks(self, data, length=500, ovlp=50, num_chunks=10, start_point=-1): 
        '''2 seconds, 0.2 seconds overlap'''
        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks
        
        if start_point == -1:
            if num_chunks * length > total_len - 1:
                start_point = 0
                # actual_num_chunks = total_len // length
            else:
                start_point = np.random.randint(0, total_len - num_chunks * length)
        
        for i in range(actual_num_chunks):
            if start_point + length > total_len:
                break
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point = start_point + length - ovlp
        return np.array(all_chunks), start_point

    
    def print_subjects(self) -> None:
        subjects_list = []
        [subjects_list.append(entry['subject']) for entry in self.data if entry['subject'] not in subjects_list]
        print("Subjects list:", subjects_list)

    def map_channels(self, eeg_signal):
        '''
        Channel mapping from NeuroGPT's TUH dataset (https://arxiv.org/pdf/2311.03764) to https://arxiv.org/pdf/2406.07151 dataset.
        According to https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
        '''
        orig_channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
        new_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'OZ', 'T1', 'T2']
        
        # filtered_eeg_signal = [eeg_signal[i] for i in range(len(orig_channels)) if orig_channels[i] in new_channels]

        new_eeg_signal = []
        for ch in new_channels:
            match ch:
                case "T1":
                    ch = "FT7"
                case "T2":
                    ch = "FT8"
                case "T3":
                    ch = "T7"
                case "T4":
                    ch = "T8"
                case "T5":
                    ch = "P7" # TODO: could match this to TP7
                case "T6":
                    ch = "P8" # TODO: could match this to TP8
            
            orig_ch_idx = orig_channels.index(ch)
            if orig_ch_idx != -1:
                new_eeg_signal.append(eeg_signal[orig_ch_idx, :])
            else:
                print("Oops! Can't map unknown channel", ch)
                return

        return torch.stack(new_eeg_signal, dim = 0)
    