from speechbrain.inference import SpeakerRecognition, SpectralMaskEnhancement
from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torchaudio
import torchaudio.transforms as T

XVECTOR_SAMPLING_RATE = 16_000
METRICGAN_SAMPLING_RATE = 16_000
SILERO_SAMPLING_RATE = 16_000
WHISPER_SAMPLING_RATE = 16_000
MAX_ROUND_SECONDS = 1.5
DEFAULT_RECORD_RATE = 44_100

class VoiceID:
    def __init__(self, config) -> None:
        # Load the Silero VAD model
        silero, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                      model='silero_vad', trust_repo=True)
        (self.get_speech_timestamps, _, read_audio, _, _) = utils
        self.silero = silero
        
        # Load the ECAPA Voiceprint model
        self.ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="./pretrained_models/spkrec"
        )
        
        # Initialize the record
        self.record: np.ndarray = np.array([]) # (channels, samples)
        self.round_cache: np.ndarray = np.array([]) # (channels, samples)
        self.rate: int = DEFAULT_RECORD_RATE
        
    def preprocess(self, samples: np.ndarray|torch.Tensor, 
                   src_rate: int = DEFAULT_RECORD_RATE, 
                   tgt_rate: int = XVECTOR_SAMPLING_RATE) -> Tensor:
        '''
        Args:
            samples: np.ndarray|torch.Tensor: The audio samples to preprocess, (channels, samples)
        '''
        # all to tensor
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples)
        
        # The T.Resample requires Double(float32)! Instead of float64
        samples = samples.to(dtype=torch.float32)
        samples = T.Resample(src_rate, tgt_rate)(samples) if src_rate != tgt_rate else samples
        # samples = T.Vad(sample_rate=tgt_rate)(samples)
        samples = samples.mean(dim=0, keepdim=True)
        return samples # (1, samples)
        
    def extract_round_features(self) -> Tensor:
        round_record = self.preprocess(self.round_cache, self.rate, METRICGAN_SAMPLING_RATE)
        return self.ecapa.encode_batch(round_record)
    
    def is_round_end(self) -> bool:
        if len(self.round_cache)//DEFAULT_RECORD_RATE >= MAX_ROUND_SECONDS:
            return True
        
        round_record = self.preprocess(self.round_cache, self.rate, SILERO_SAMPLING_RATE)
        timestamps = self.get_speech_timestamps(round_record, 
                        self.silero, 
                        sampling_rate=SILERO_SAMPLING_RATE,
                        threshold=0.1)
        return len(timestamps) > 0
    
    def add_chunk(self, chunk: np.ndarray, rate: int=DEFAULT_RECORD_RATE) -> None:
        '''
        Args:
            chunk: np.ndarray: The audio samples to add, (channels, samples)
        '''
        if len(self.record) == 0:
            self.record = chunk
            self.round_cache = chunk
            self.rate = rate
        else:
            self.record = np.concatenate([self.record, chunk[1]])
            self.round_cache = np.concatenate([self.round_cache, chunk[1]])
            assert self.rate == rate, "The rate of the chunk does not match the rate of the record!"
        
    def load_record(self, record: np.ndarray, rate: int) -> None:
        '''
        Args:
            record: np.ndarray: The audio samples to load, (channels, samples)
        '''
        self.record = record
        self.rate = rate
        
    def enhance(self, record: np.ndarray, rate: int) -> np.ndarray:
        '''
        Args:
            record: np.ndarray: The audio samples to enhance, (channels, samples)
        '''
        record = self.preprocess(record, rate, METRICGAN_SAMPLING_RATE)
        length = torch.ones(record.shape[0])
        return self.metircgan.enhance_batch(record, lengths=length)

    def get_round_slices(self, 
                         record: torch.Tensor|None = None, 
                         rate: int = DEFAULT_RECORD_RATE) -> list[torch.Tensor]:
        record = self.record if record is None else record
        record = self.preprocess(record, rate, SILERO_SAMPLING_RATE)
        timestamps = self.get_speech_timestamps(record, 
                        self.silero, 
                        sampling_rate=SILERO_SAMPLING_RATE,
                        threshold=0.2, return_seconds=True)
        slices = [record[:, int(stamp['start']*SILERO_SAMPLING_RATE):int(stamp['end']*SILERO_SAMPLING_RATE)]
                  for stamp in timestamps]
        return slices
    
    def extract_clip_features(self, 
                              record: torch.Tensor|None = None, 
                              rate: int = DEFAULT_RECORD_RATE) -> torch.Tensor:
        record = self.record if record is None else record
        slices = self.get_round_slices(record, rate) # [(1, samples), ...]
        slices = [slice.squeeze(0)[len(slice)//2:] for slice in slices] # [(samples,) ...]
        lengths = torch.tensor([slice.shape[0] for slice in slices]) # (batch,)
        slices = pad_sequence(slices, batch_first=True) # (batch, samples)
        print(slices.shape)
        return self.ecapa.encode_batch(slices, lengths) # (batch, channels, emb_dim)