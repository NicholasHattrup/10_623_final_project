from torch.utils.data import Dataset
import os


class MoleculeDataset(Dataset):
    
    def __init__(
            self,
            file_path : os.PathLike
    ) -> None:
        
        with open(file_path, "r") as f:
            #TODO PARSE DATA

        self.smiles_strs = 
        self.smiles_coords = 
        self.quantum_coords = 

    def __len__(self):
        return len(self.smiles_strs)
    
    def __getitem__(self, idx: int):
        return self.smiles_coords[idx], self.quantum_coords[idx] 



# class ClusteredHDF5Dataset(Dataset):
#     def __init__(
#         self,
#         file_path: Path | str,
#         clusters: list[SequenceCluster],
#         label_type : str
#     ) -> None:
#         """A PyTorch Dataset for loading data from an HDF5 file according to any sampling strategy.
#            This returns the untokenized protein and its associated label

#         Parameters
#         ----------
#         file_path : Path | str
#             The path to the HDF5 file containing the data (keys are protein tags)
#         clusters : list[SequenceCluster]
#             A list of sequences clustered with MMSeqs. Contains the tag and sequence data
#         label_type : str
#             Which label to expect in the dataset (e.g. distogram)
#         """
#         self.file_path = file_path
#         self.clusters = clusters
#         self.label_type = label_type

#     @property
#     def h5_data(self) -> h5py.File:
#         """Lazy load the h5 file in the dataloader worker process."""
#         if not hasattr(self, "_h5_data"):
#             self._h5_data = h5py.File(self.file_path, "r") # should be closed by upon garbage collection
#         return self._h5_data

#     def __len__(self) -> int:
#         return len(self.clusters)

#     def __getitem__(self, idx: int) -> Dict[str, str]:
#         # Get random sample from the idx'th cluster
#         seq_like = np.random.choice(self.clusters[idx])

#         group = self._h5_data[seq_like.tag]
#         # seq = group["sequence"][()] # should be same as seq_like.sequence

#         label = torch.from_numpy(group[self.label_type][:])

#         prot = Protein(tag = seq_like.tag, sequence = seq_like.sequence, 
#                         disorder = group["iupred3_disorder_score"][:],
#                         disorder_binding = group["anchor2_disorder_binding_score"][:], 
#                         hydropathy = group["hydropathy"][:]
#                     )

#         return prot, label
