import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from .base import *

# TODO: refactor, reuse code from .base
class Alljoined1(Dataset):
    def __init__(self, data, labels, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, population_mean=0, population_std=1, gpt_only=False, normalization=True, start_samp_pnt=-1):

        print("Loading the data...")
        self.data = data
        self.labels = labels
        
        print(type(self.data))
        print(self.data.shape)
        print(self.data[0].keys())
        print(type(self.data[0]['EEG']))
        print(len(self.data[0]['EEG']))
        print(type(self.data[0]['EEG'][0]))
        print(len(self.data[0]['EEG'][0]))

        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only=gpt_only
        self.start_samp_pnt = start_samp_pnt

        
    def __getitem__(self, index):
        data = self.data[index]['EEG']
        image_id = self.data[index]['73k_id']

        # reorder channels
        data = self.reorder_channels(data)
        
        return self.preprocess_sample(data, seq_len=self.num_chunks, labels=self.labels[image_id])

    def __len__(self):
        return len(self.data)

    def reorder_channels(self, data):
        orig_eegimagenet_ch_order = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
        alljoined1_ch_order = ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'] # removed 'Iz' and 'Status' channels
        eegimagenet_channels_missing_from_alljoined = ['PO5', 'PO6', 'CB1', 'CB2']

        # remove EEGImageNet channels that are not in Alljoined1
        [orig_eegimagenet_ch_order.remove(ch) for ch in eegimagenet_channels_missing_from_alljoined]
                
        reordered = np.zeros((len(orig_eegimagenet_ch_order), len(data[0])),)
        for orig_idx, ch_name in enumerate(orig_eegimagenet_ch_order):
            mapped_idx = alljoined1_ch_order.index(ch_name)
            if mapped_idx != -1:    # the EEGImageNet channel doesn't exist in Alljoined
                reordered[orig_idx, :] = data[mapped_idx][:]
            else:
                print("Error (reorder_channels): couldn't find channel!")

        return reordered

    def find_uncommon_channels(self) -> None:
        eegimagenet_ch_order = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
        alljoined1_ch_order = ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'Status']
        
        uncommon_channels = []
        for ch in orig_eegimagenet_ch_order:
            if ch not in alljoined1_ch_order:
                uncommon_channels.append(ch)

        print(uncommon_channels)

        uncommon_channels = []
        for ch in alljoined1_ch_order:
            if ch.upper() not in orig_eegimagenet_ch_order:
                uncommon_channels.append(ch)

        print(uncommon_channels)

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

    def preprocess_sample(
        self,
        sample,
        seq_len,
        labels=None
        ) -> Dict[str, torch.Tensor]:
        out = {}
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len, self.start_samp_pnt)

        attention_mask = np.ones(seq_len)
        chunks = EEGDataset._pad_seq_right_to_n(
            seq=chunks,
            n=seq_len,
            pad_value=0
        )

        attention_mask = EEGDataset._pad_seq_right_to_n(
            seq=attention_mask, 
            n=seq_len,
            pad_value=0
        )
        
        if self.gpt_only == True:
            chunks = np.reshape(chunks, (seq_len, chunks.shape[1]*chunks.shape[2]))
        out["inputs"] = torch.from_numpy(chunks).to(torch.float)
        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len
        
        if self.sample_keys is not None:
            out = {
                key: out[key] 
                for key in self.sample_keys
                if key in out
            }

        if labels is not None:
            out['labels'] = torch.from_numpy(np.array(labels)).to(torch.float32)
   
        return out
    
    def normalize(self, data):
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        # Ensure std is not zero to avoid division by zero.
        # If std is zero, normalization doesn't make sense, 
        # so you might set std to a small positive value or handle it in another way.
        # std = np.where(std == 0, 1e-23, std)
        return (data - mean) / (std + 1e-25)
    


def onehot_encode(categories_list: list) -> dict[str, list]:
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(categories_list)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    onehot_encoded_dict = {}
    for el in onehot_encoded:
        category_str = label_encoder.inverse_transform([np.argmax(el[:])])
        onehot_encoded_dict[category_str[0]] = el
    return onehot_encoded_dict


def load_alljoined1_dataset(config: Dict) -> tuple[Dataset, Dataset, Dataset]:
    ds = datasets.load_dataset("Alljoined/05_125") # HuggingFace

    # process the labels
    #labels = []
    #labels_coco = []
    #[labels.append(entry['73k_id']) for entry in ds['train'] if entry['73k_id'] not in labels]
    with open("../Datasets/Alljoined1/captions_and_categories.json") as f:
        labels_json = json.load(f)

    # make a list of all supercategories
    categories = []
    labels = {}
    for entry in labels_json:
        image_categories_list = entry["categories"] # the categories an image belongs to
        image_cats = [cat["supercategory_name"] for cat in image_categories_list]
        categories.extend(image_cats)
        labels[int(entry["nsdId"]) - 1] = set(image_cats) # in the HuggingFace dataset the NSD ID (73k_id) is saved as nsd_id - 1, so I have to subtract 1 here as well, to match IDs

    # eliminate the duplicates
    categories = list(set(categories))
    onehot_encoding = onehot_encode(categories)

    # make dict = {image_id: onehot_encoding}
    for image_id, cats in labels.items():
        labels[image_id] = sum([onehot_encoding[cat] for cat in cats])

    # make a dict {"supercategory": one-hot encoded supercategory}
    #encoded_cat = pd.get_dummies(pd.DataFrame(data={'col1':categories})['col1'])
    
    #onehot_encoder = OneHotEncoder(sparse_output=False)
    #onehot_encoded = onehot_encoder.fit_transform(categories)
    #print(encoded_cat)

    #labels[entry["nsId"]] = entry["categories"]
    
    train_data = Alljoined1(ds['train'], labels, sample_keys=['inputs', 'attention_mask'], chunk_len=config["chunk_len"], num_chunks=config["num_chunks"],
                                         ovlp=config["chunk_ovlp"], gpt_only= not config["use_encoder"])
    test_data = Alljoined1(ds['test'], labels, sample_keys=['inputs', 'attention_mask'], chunk_len=config["chunk_len"], num_chunks=config["num_chunks"],
                                       ovlp=config["chunk_ovlp"], gpt_only= not config["use_encoder"])

    return train_data, None, test_data