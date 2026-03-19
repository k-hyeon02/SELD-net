import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cls_feature_class


class SELDDataset(Dataset):
    """
    PyTorch Dataset for SELDnet

    전처리된 .npy 피처/레이블 파일을 읽어 seq_len 길이의 시퀀스 단위로 반환
    DataLoader와 함께 사용하면 배치, 셔플, 병렬 로딩을 자동으로 처리

    반환 샘플:
        feat      : (2*nb_ch, seq_len, feat_len)  - 스펙트로그램 (channels-first)
        sed_label : (seq_len, nb_classes)          - 음원 존재 여부 (0 or 1)
        doa_label : (seq_len, nb_classes*2 or *3)  - 방향각 직교 좌표 (x,y) or (x,y,z)
    """

    def __init__(
            self, datagen_mode='train', dataset='ansim', ov=1, split=1, db=30,
            seq_len=64, nfft=512, classifier_mode='regr', weakness=0,
            xyz_def_zero=False, extra_name='', azi_only=False
    ):
        """
        1. FeatureClass로 피처/레이블 폴더 경로 설정
        2. 'train' or 'test' 파일만 필터링 → filenames_list
        3. 첫 파일 열어서 shape 파악 (nb_frames_file, feat_len, label_len)
        4. seqs_per_file = nb_frames_file // seq_len
        5. 캐시 딕셔너리 초기화
        
        Args:
        datagen_mode  : 'train' 또는 'test' - 해당 모드의 파일만 로드
        dataset       : 데이터셋 이름 (ansim, resim, real 등)
        ov            : 최대 동시 겹치는 음원 수 (1, 2, 3)
        split         : 교차검증 split 번호
        db            : 신호 대 잡음비
        seq_len       : 한 시퀀스의 프레임 수 (모델 입력 길이)
        nfft          : FFT 크기
        classifier_mode: 레이블 형식 ('regr'만 지원)
        weakness      : 레이블 약화 정도 (regr에서는 미사용)
        xyz_def_zero  : 음원 없는 프레임의 DOA를 (0,0,0)으로 설정할지 여부
        azi_only      : True면 DOA를 (x,y)만, False면 (x,y,z)로 반환
        extra_name    : 폴더명에 붙는 추가 문자열
        """
        self._datagen_mode = datagen_mode
        self._seq_len = seq_len        # 한 시퀀스의 프레임 수
        self._xyz_def_zero = xyz_def_zero
        self._azi_only = azi_only

        # cls_feature_class를 통해 피처/레이블 폴더 경로 가져오기
        self._feat_cls = cls_feature_class.FeatureClass(
            dataset=dataset, ov=ov, split=split, db=db, nfft=nfft
        )
        self._label_dir = self._feat_cls.get_label_dir(classifier_mode, weakness, extra_name)
        self._feat_dir = self._feat_cls.get_normalized_feat_dir(extra_name)

        self._2_nb_ch = 2 * self._feat_cls.get_nb_channels()  # 진폭+위상 채널 수 (예: 4채널 마이크 → 8)
        self._nb_classes = len(self._feat_cls.get_classes())   # 음원 클래스 수
        self._default_azi, self._default_ele = self._feat_cls.get_default_azi_ele_regr()  # 음원 없을 때 기본 각도

        # datagen_mode('train' 또는 'test')가 파일명에 포함된 파일만 필터링
        self._filenames_list = sorted([
            f for f in os.listdir(self._label_dir) if datagen_mode in f
        ])

        # 첫 번째 파일을 열어 shape 정보 파악
        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
        self._nb_frames_file = temp_feat.shape[0]              # 파일당 프레임 수 (예: 5166)
        self._feat_len = temp_feat.shape[1] // self._2_nb_ch   # 주파수 빈 수 (예: 256)

        temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
        self._label_len = temp_label.shape[-1]  # 레이블 전체 길이 = nb_classes(SED) + nb_classes(azi) + nb_classes(ele)

        # 파일 하나를 seq_len 프레임씩 나누면 몇 개의 시퀀스가 되는지
        # 예: 5166 frames / 64 seq_len = 80 sequences (나머지는 버림)
        self._seqs_per_file = self._nb_frames_file // self._seq_len

        # 파일 캐시: 같은 파일을 __getitem__ 호출마다 디스크에서 다시 읽지 않도록 메모리에 저장
        self._cache_feat = {}
        self._cache_label = {}

        print(
            'Mode: {}, files: {}, classes: {}\n'
            'frames/file: {}, feat_len: {}, nb_ch: {}, label_len: {}\n'
            'seqs/file: {}, total_seqs: {}'.format(
                datagen_mode, len(self._filenames_list), self._nb_classes,
                self._nb_frames_file, self._feat_len, self._2_nb_ch, self._label_len,
                self._seqs_per_file, len(self)
            )
        )

    def __len__(self):
        # DataLoader가 전체 샘플 수를 알기 위해 호출
        # 전체 시퀀스 수 = 파일 수 × 파일당 시퀀스 수
        return len(self._filenames_list) * self._seqs_per_file

    def __getitem__(self, idx):
        """
        DataLoader가 idx를 넘기면 해당 시퀀스 하나를 반환
        idx → 몇 번째 파일의 몇 번째 시퀀스인지 계산

        idx
         ↓
        file_idx = idx // seqs_per_file  → 몇 번째 파일?
        seq_idx  = idx % seqs_per_file   → 그 파일의 몇 번째 시퀀스?
         ↓
        캐시 확인 → 없으면 .npy 로드 후 캐시 저장
         ↓
        start = seq_idx * seq_len
        end   = start + seq_len
         ↓
        feat[start:end]  → reshape → transpose → FloatTensor
        label[start:end] → SED 분리 / DOA 직교 좌표 변환 → FloatTensor
         ↓
        (feat, sed_label, doa_label) 반환
        """
        file_idx = idx // self._seqs_per_file  # 파일 인덱스
        seq_idx  = idx % self._seqs_per_file   # 해당 파일 내 시퀀스 인덱스

        filename = self._filenames_list[file_idx]

        # 캐시에 없으면 디스크에서 로드 후 캐시에 저장
        if filename not in self._cache_feat:
            self._cache_feat[filename] = np.load(os.path.join(self._feat_dir, filename))
            self._cache_label[filename] = np.load(os.path.join(self._label_dir, filename))

        feat  = self._cache_feat[filename]   # (nb_frames_file, feat_len * 2*nb_ch)
        label = self._cache_label[filename]  # (nb_frames_file, label_len)

        # 해당 시퀀스의 프레임 구간 계산
        start = seq_idx * self._seq_len
        end   = start + self._seq_len

        # --- 피처 변환 ---
        feat_seq = feat[start:end, :]  # (seq_len, feat_len * 2*nb_ch)
        feat_seq = feat_seq.reshape(self._seq_len, self._feat_len, self._2_nb_ch)  # → (seq_len, feat_len, 2*nb_ch)
        feat_seq = np.transpose(feat_seq, (2, 0, 1))  # → (2*nb_ch, seq_len, feat_len) : PyTorch channels-first

        # --- 레이블 분리 ---
        label_seq = label[start:end, :]  # (seq_len, label_len)
        # label_len 구조: [SED(nb_classes) | azi(nb_classes) | ele(nb_classes)]

        # SED 레이블: 앞 nb_classes 열 → 프레임별 음원 존재 여부 (0 or 1)
        sed_label = label_seq[:, :self._nb_classes]  # (seq_len, nb_classes)

        # --- DOA 레이블: 각도(azi/ele) → 직교 좌표(x,y,z) 변환 ---
        # 각도 대신 직교 좌표를 쓰는 이유:
        # 359°와 1°는 실제로 2° 차이지만 값의 차이는 358 → 신경망이 학습하기 어려움
        # 직교 좌표로 변환하면 이런 불연속성이 없어짐
        if self._azi_only:
            # azimuth만 사용: (x, y) = (cos(azi), sin(azi))
            azi_rad = label_seq[:, self._nb_classes:2*self._nb_classes] * np.pi / 180
            x = np.cos(azi_rad)
            y = np.sin(azi_rad)

            # 음원이 없는 프레임의 DOA를 (0, 0)으로 설정
            if self._xyz_def_zero:
                no_sound = np.where(label_seq[:, 2*self._nb_classes:] == self._default_ele)
                x[no_sound] = 0
                y[no_sound] = 0

            doa_label = np.concatenate((x, y), axis=-1)  # (seq_len, nb_classes*2)
        else:
            # azimuth + elevation 모두 사용: sph2cart 변환과 동일
            azi_rad = label_seq[:, self._nb_classes:2*self._nb_classes] * np.pi / 180
            ele_rad = label_seq[:, 2*self._nb_classes:] * np.pi / 180
            cos_ele = np.cos(ele_rad)

            x = np.cos(azi_rad) * cos_ele  # 수평 투영 후 x 성분
            y = np.sin(azi_rad) * cos_ele  # 수평 투영 후 y 성분
            z = np.sin(ele_rad)            # 수직(높이) 성분

            # 음원이 없는 프레임의 DOA를 (0, 0, 0)으로 설정
            if self._xyz_def_zero:
                no_sound = np.where(label_seq[:, 2*self._nb_classes:] == self._default_ele)
                x[no_sound] = 0
                y[no_sound] = 0
                z[no_sound] = 0

            doa_label = np.concatenate((x, y, z), axis=-1)  # (seq_len, nb_classes*3)

        # numpy → PyTorch Tensor로 변환
        return (
            torch.FloatTensor(feat_seq),    # (2*nb_ch, seq_len, feat_len)
            torch.FloatTensor(sed_label),   # (seq_len, nb_classes)
            torch.FloatTensor(doa_label)    # (seq_len, nb_classes*2 or *3)
        )

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_data_sizes(self):
        """배치 제외한 단일 샘플의 shape 반환"""
        doa_dim = 2 if self._azi_only else 3
        feat_shape = (self._2_nb_ch, self._seq_len, self._feat_len)
        label_shape = [
            (self._seq_len, self._nb_classes),
            (self._seq_len, self._nb_classes * doa_dim)
        ]
        return feat_shape, label_shape
