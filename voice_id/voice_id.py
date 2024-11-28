from typing import TypeAlias
from speechbrain.inference import SpeakerRecognition
from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from config import Config
from .utils import resample, cancel_channel, add_channel, to_numpy, to_tensor
from scipy.signal import hilbert

Audio: TypeAlias = tuple[int, np.ndarray|torch.Tensor]

XVECTOR_SAMPLING_RATE = 16_000
METRICGAN_SAMPLING_RATE = 16_000
SILERO_SAMPLING_RATE = 16_000
WHISPER_SAMPLING_RATE = 16_000
ECAPA_SAMPLING_RATE = 16_000
DEFAULT_RECORD_RATE = 16_000

class VoiceID:
    def __init__(self, config: Config) -> None:
        self.round_threshold = config.voice_round_threshold
        self.video_threshold = config.voice_video_threshold
        self.max_round_seconds = config.voice_max_round_seconds
        
        # Load the Silero VAD model
        silero, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                      model='silero_vad', trust_repo=True)
        self.get_speech_timestamps, _, _, _, _ = utils
        self.silero = silero
        
        # Load the ECAPA Voiceprint model
        self.ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        
        # Initialize the record
        self.record: np.ndarray = np.array([]) # (channels, samples)
        self.round_cache: np.ndarray = np.array([]) # (channels, samples)
        self.last_round_cache: np.ndarray = np.array([]) # (channels, samples)
        self.last_is_end = False
        
    def extract_round_features(self) -> Tensor:
        '''
        Returns:
            Tensor: The extracted features, (1, emb_dim)
        '''
        if self.last_round_cache.size == 0:
            return None
        else:
            features = self.extract_label_features((DEFAULT_RECORD_RATE, self.last_round_cache))
            self.last_round_cache = np.array([])
            return features
    
    def extract_label_features(self, label_audio: Audio) -> Tensor:
        '''
        Args:
            label_audio: Audio: The audio samples to extract features from, (rate, wave)
        Returns:
            Tensor: The extracted features, (emb_dim, )
        '''
        rate, wave = label_audio
        wave = to_tensor(wave)
        wave = resample(wave, rate, ECAPA_SAMPLING_RATE)
        wave = add_channel(wave) # (1, samples)
        features = self.ecapa.encode_batch(wave).squeeze()
        return features
    
    def is_round_end(self) -> bool:
        """
        Determines if the current round has ended based on the round cache.
        Returns:
            bool: True if the round has ended, False otherwise.
        """
        if len(self.round_cache) // DEFAULT_RECORD_RATE >= self.max_round_seconds:
            print(f"检测到语音超时，当前长度: {self.round_cache.shape}")
            self.last_is_end = False
            self.last_round_cache = np.array([])
            self.round_cache = np.array([])
            return True
        
        envelope = np.abs(hilbert(self.round_cache))
        this_is_end = np.max(envelope) > self.round_threshold
        is_end = self.last_is_end
        self.last_is_end = (not self.last_is_end) and this_is_end
        
        if is_end:
            print(f"检测到语音结束，当前长度: {self.round_cache.shape}")
            self.last_round_cache = self.round_cache.copy()
            self.round_cache = np.array([])
        else:
            print(f"未检测到语音结束，当前长度: {self.round_cache.shape}")
        
        return is_end
    
    def add_chunk(self, chunk: Audio) -> None:
        '''
        Args:
            chunk: Audio: The audio chunk to add, (rate, wave)=(int, np.ndarray)
        '''
        rate, wave = chunk
        wave = resample(wave, rate, DEFAULT_RECORD_RATE)
        wave = cancel_channel(wave)
        if len(self.record) == 0:
            self.record = wave
            self.round_cache = wave
        else:
            self.record = np.concatenate([self.record, wave], axis=-1)
            self.round_cache = np.concatenate([self.round_cache, wave], axis=-1)
        
    def load_record(self, record: Audio) -> None:
        '''
        Args:
            record: Audio: The audio samples to load, (rate, wave)
        '''
        rate, wave = record
        wave = resample(wave, rate, DEFAULT_RECORD_RATE)
        wave = cancel_channel(wave)
        wave = to_numpy(wave)
        self.record = wave

    def get_round_slices(self, record: Audio = None) -> list[torch.Tensor]:
        '''
        Args:
            record: Audio: The audio samples to extract slices from, (rate, wave)
        Returns:
            list[torch.Tensor]: The extracted slices, [(1, samples) ...]
        '''
        rate, wave = record if record is not None else (DEFAULT_RECORD_RATE, self.record)
        wave = to_tensor(wave)
        wave = resample(wave, rate, SILERO_SAMPLING_RATE)
        wave = add_channel(wave)
        timestamps = self.get_speech_timestamps(wave, 
                        self.silero, 
                        sampling_rate=SILERO_SAMPLING_RATE,
                        threshold=self.video_threshold, return_seconds=True)
        slices = [wave[:, int(stamp['start']*SILERO_SAMPLING_RATE):int(stamp['end']*SILERO_SAMPLING_RATE)]
                  for stamp in timestamps]
        # for i, slice in enumerate(slices):
        #     torchaudio.save(f'audio/slice_{i}.wav', slice.cpu(), SILERO_SAMPLING_RATE)
        return slices
    
    def extract_clip_features(self, record: Audio) -> torch.Tensor:
        """
        Extracts features from an audio clip.
        Args:
            record (Audio): An audio recording of a clip from which features are to be extracted.
        Returns:
            torch.Tensor: A tensor containing the extracted features. The shape of the tensor is 
                          (batch, channels, emb_dim). If no slices are found, an empty list is returned.
        """
        slices = self.get_round_slices(record)
        slices = [slice.squeeze(0)[len(slice)//2:] for slice in slices] # [(samples,) ...]
        lengths = torch.tensor([slice.shape[0] for slice in slices]) # (batch,)
        if len(lengths)==0:
            return []
        slices = pad_sequence(slices, batch_first=True) # (batch, samples)
        return self.ecapa.encode_batch(slices, lengths).squeeze(1) # (batch, channels, emb_dim)
    